"""
Multi-agent GRPO training for the improv game "Converge" — chain-of-thought variant.

Identical to grpo_converge_optimized.py except the model generates a brief
chain-of-thought before its final word.  GRPO trains only on the word token(s);
the CoT prefix is included in the prompt context but excluded from the loss.

Generation format:
    Thinking: [brief reasoning, up to ~cot_tokens tokens]
    Word: [word]

Usage:
    # Smoke test (CPU, no training)
    python grpo_converge_cot.py --eval_only --eval_episodes 5

    # Full training (single GPU, LoRA)
    python grpo_converge_cot.py --use_peft

    # Two different base checkpoints
    python grpo_converge_cot.py --model_a Qwen/Qwen2.5-3B-Instruct \
                                 --model_b Qwen/Qwen2.5-3B-Instruct \
                                 --use_peft
"""

import re
import random
import argparse
import string
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoTokenizer

from grpo import Rollout, GRPOLoop, make_policy, make_ref, make_dual_adapter_policy

try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

# Checking bridge validity

def _best_path_sim(w1: str, w2: str) -> float:
    syns1 = wn.synsets(w1, pos=wn.NOUN)
    syns2 = wn.synsets(w2, pos=wn.NOUN)
    if not syns1 or not syns2:
        return 0.0
    best = 0.0
    for s1 in syns1[:3]:
        for s2 in syns2[:3]:
            sim = s1.path_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best

def _best_wup_sim(w1: str, w2: str) -> float:
    syns1 = wn.synsets(w1, pos=wn.NOUN)
    syns2 = wn.synsets(w2, pos=wn.NOUN)
    if not syns1 or not syns2:
        return 0.0
    best = 0.0
    for s1 in syns1[:3]:
        for s2 in syns2[:3]:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best

def _best_sim(w1: str, w2: str) -> float:
    return _best_wup_sim(w1, w2)

def is_valid_bridge(word: str, anchor_a: str, anchor_b: str,
                    threshold: float = 0.10) -> bool:
    sim_a = _best_sim(word, anchor_a)
    sim_b = _best_sim(word, anchor_b)
    return sim_a >= threshold and sim_b >= threshold

def is_in_wordnet(word: str) -> bool:
    return bool(wn.synsets(word, pos=wn.NOUN))

# Make sure words are concrete
CONCRETE_LEXNAMES = {
    "noun.animal", "noun.plant", "noun.food", "noun.artifact",
    "noun.body", "noun.object", "noun.substance",
}

_CONCRETE_NOUN_CACHE: list[str] = []

def get_concrete_nouns(max_words: int = 3000) -> list[str]:
    global _CONCRETE_NOUN_CACHE
    if _CONCRETE_NOUN_CACHE:
        return _CONCRETE_NOUN_CACHE

    min_freq = 2
    try:
        from nltk.corpus import brown
        freq = {}
        for w in brown.words():
            w = w.lower()
            freq[w] = freq.get(w, 0) + 1
        common_words = {w for w, c in freq.items() if c >= min_freq}
        print(f"[WordNet] Brown corpus loaded: {len(common_words)} common words.")
    except Exception:
        common_words = None
        print("[WordNet] Brown corpus unavailable — no frequency filter applied.")

    seen = set()
    words = []
    for synset in wn.all_synsets(pos=wn.NOUN):
        if synset.lexname() not in CONCRETE_LEXNAMES:
            continue
        for lemma in synset.lemmas():
            w = lemma.name().lower().replace("_", "")
            if (w.isalpha() and 3 <= len(w) <= 12
                    and " " not in w and w not in seen
                    and (common_words is None or w in common_words)):
                seen.add(w)
                words.append(w)
                if len(words) >= max_words:
                    break
        if len(words) >= max_words:
            break

    _CONCRETE_NOUN_CACHE = words
    print(f"[WordNet] Loaded {len(words)} concrete nouns.")
    return words

# Prompt

SYSTEM_PROMPT = """You are playing a word convergence game with a partner.

Each round, both you and your partner say ONE word. The word must be
semantically related to BOTH words from the previous round — it should
bridge or connect them. The goal is for both players to eventually say
the EXACT SAME word in the same round.

Rules:
- The word must be a real English noun.
- The word must be meaningfully related to both words from the last round.
- You win (and score more points) the sooner you both say the same word.

Output format (two lines, exactly):
Thinking: [max 8 words]
Word: [single lowercase word]"""

def build_messages(anchor_a: str, anchor_b: str,
                   history: list[dict], agent_label: str,
                   start_a: str = "", start_b: str = "") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not history:
        user_content = (
            f"Game start!\n"
            f"The two starting words are: '{anchor_a}' and '{anchor_b}'\n\n"
            f"Round 1: Reason briefly (max 8 words), then give your word:\n"
            f"Thinking: [max 8 words]\nWord: [word]"
        )
    else:
        history_lines = []
        for i, h in enumerate(history):
            history_lines.append(
                f"  Round {i+1}: words were '{h['word_a']}' and '{h['word_b']}'"
            )
        history_str = "\n".join(history_lines)
        user_content = (
            f"Starting words: '{start_a}' and '{start_b}'\n\n"
            f"History:\n{history_str}\n\n"
            f"Last round: '{anchor_a}' and '{anchor_b}'.\n"
            f"Round {len(history) + 1}: Reason briefly (max 8 words), then give your word:\n"
            f"Thinking: [max 8 words]\nWord: [word]"
        )

    messages.append({"role": "user", "content": user_content})
    return messages

# Fast (non-CoT) prompt used for the actual training rollouts every step.
# GRPO never puts gradient on the CoT tokens anyway (see plays_to_rollout),
# so training rollouts skip the "Thinking:" preamble entirely — this is the
# same short single-word format as grpo_converge_optimized.py. The CoT
# format above is reserved for the periodic side-eval (see --cot_eval_every)
# that tracks how reasoning-conditioned play evolves without slowing down
# every training step by ~10x (80 CoT tokens vs 8 word-only tokens).

SYSTEM_PROMPT_FAST = """You are playing a word convergence game with a partner.

Each round, both you and your partner say ONE word. The word must be
semantically related to BOTH words from the previous round — it should
bridge or connect them. The goal is for both players to eventually say
the EXACT SAME word in the same round.

Rules:
- Output EXACTLY ONE word. No punctuation, no explanation, no other text.
- The word must be a real English noun.
- The word must be meaningfully related to both words from the last round.
- You win (and score more points) the sooner you both say the same word.

Output format: a single lowercase word and nothing else.
Example of CORRECT output: river
Example of INCORRECT output: I think the word is river."""

def build_messages_fast(anchor_a: str, anchor_b: str,
                        history: list[dict], agent_label: str,
                        start_a: str = "", start_b: str = "") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_FAST}]

    if not history:
        user_content = (
            f"Game start!\n"
            f"The two starting words are: '{anchor_a}' and '{anchor_b}'\n\n"
            f"Round 1: Output ONE word that meaningfully connects BOTH of these words."
        )
    else:
        history_lines = []
        for i, h in enumerate(history):
            history_lines.append(
                f"  Round {i+1}: words were '{h['word_a']}' and '{h['word_b']}'"
            )
        history_str = "\n".join(history_lines)
        user_content = (
            f"Starting words: '{start_a}' and '{start_b}'\n\n"
            f"History:\n{history_str}\n\n"
            f"Last round: '{anchor_a}' and '{anchor_b}'.\n"
            f"Round {len(history) + 1}: Output ONE word that meaningfully "
            f"connects BOTH '{anchor_a}' and '{anchor_b}'."
        )

    messages.append({"role": "user", "content": user_content})
    return messages

# ---------------------------------------------------------------------------
# 4.  Single-word extraction & cleaning
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b([a-zA-Z]{2,15})\b")

_PREAMBLE_WORDS = {
    "i", "my", "the", "a", "an", "is", "think", "word", "would", "be",
    "say", "said", "answer", "output", "bridge", "related", "both",
    "guess", "round", "game", "player", "partner", "correct", "final",
    "response", "choose", "pick", "select", "go", "okay", "ok", "sure",
    "well", "so", "then", "thus", "hence", "therefore", "let", "me",
    "give", "here", "this", "that", "it", "its",
}


def extract_word(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    else:
        return None

    text = text.strip(string.punctuation + string.whitespace).lower()

    if re.fullmatch(r"[a-z]{2,15}", text):
        return text

    tokens = _WORD_RE.findall(text)
    for tok in tokens:
        tok = tok.lower()
        if tok not in _PREAMBLE_WORDS and len(tok) >= 2:
            return tok

    return None


def sanitize_output(raw: str) -> Optional[str]:
    return extract_word(raw)


def parse_cot_output(raw: str) -> tuple[str, str]:
    """
    Split CoT output into (cot_prefix, word_str).

    Looks for "Word: <word>" in raw. If found:
      - cot_prefix = everything up to and including "Word: " (used as the
                     additional prompt context so GRPO trains only on word tokens)
      - word_str   = text after "Word: " (passed to sanitize_output)
    If not found, returns ("", raw) so the full output is treated as the word.
    """
    match = re.search(r"[Ww]ord\s*:\s*", raw)
    if match:
        return raw[:match.end()], raw[match.end():]
    return "", raw


# ---------------------------------------------------------------------------
# 5.  Constrained generation
# ---------------------------------------------------------------------------

RETRY_PENALTY = 1.0
MAX_RETRIES   = 10


def _generate_cot_once(model, tokenizer, messages: list[dict],
                       device: str, temperature: float = 0.9,
                       max_new_tokens: int = 80,
                       adapter_name: "str | None" = None) -> tuple[str, str]:
    """Generate a full CoT + word response. Returns (raw_full, prompt_text)."""
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(m["content"])
            elif m["role"] == "user":
                parts.append(f"\nUser: {m['content']}\nAssistant:")
        prompt_text = "\n".join(parts)

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=768
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-3),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, prompt_text


def get_valid_word(model, tokenizer, messages: list[dict],
                   device: str, temperature: float,
                   prev_a: str, prev_b: str,
                   used_words: set[str],
                   used_stems: set[str],
                   stem_fn,
                   bridge_threshold: float,
                   cot_tokens: int = 80,
                   adapter_name: "str | None" = None) -> tuple[str, str, float, list[tuple[str,str,str]], bool]:
    """
    Returns (word, prompt_text, penalty, all_attempts, exhausted).
    all_attempts entries: (train_prompt, word_str, raw_full)
      train_prompt = original prompt + CoT prefix up to "Word: "
      word_str     = text after "Word: " (passed to sanitize_output)
      raw_full     = full generation for logging
    """
    all_attempts = []
    penalty = 0.0

    for attempt_idx in range(MAX_RETRIES):
        raw_full, prompt_text = _generate_cot_once(
            model, tokenizer, messages, device, temperature,
            max_new_tokens=cot_tokens, adapter_name=adapter_name,
        )
        cot_prefix, word_str = parse_cot_output(raw_full)
        train_prompt = prompt_text + cot_prefix
        all_attempts.append((train_prompt, word_str, raw_full))

        word = sanitize_output(word_str)

        if word is None:
            penalty += RETRY_PENALTY
            continue

        if not is_in_wordnet(word):
            penalty += RETRY_PENALTY
            continue

        if word in used_words or stem_fn(word) in used_stems:
            penalty += RETRY_PENALTY
            continue

        if not is_valid_bridge(word, prev_a, prev_b, bridge_threshold):
            penalty += RETRY_PENALTY
            continue

        return word, prompt_text, penalty, all_attempts, False

    last_train_prompt, last_word_str, _ = all_attempts[-1]
    last_word = sanitize_output(last_word_str) or "unknown"
    return last_word, last_train_prompt, penalty, all_attempts, True


def _generate_cot_batch(model, tokenizer, messages_list: list[list[dict]],
                        device: str, temperature: float = 0.9,
                        max_new_tokens: int = 80,
                        adapter_name: "str | None" = None) -> list[tuple[str, str]]:
    """Batch-generate CoT completions for a list of message lists simultaneously."""
    if not messages_list:
        return []
    if adapter_name is not None:
        model.set_adapter(adapter_name)

    prompts = []
    for messages in messages_list:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(m["content"])
                elif m["role"] == "user":
                    parts.append(f"\nUser: {m['content']}\nAssistant:")
            prompt_text = "\n".join(parts)
        prompts.append(prompt_text)

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=768,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-3),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = enc["input_ids"].shape[1]
    return [
        (prompts[i], tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip())
        for i in range(len(prompts))
    ]


def get_valid_words_batched(model, tokenizer, messages_list: list[list[dict]],
                            device: str, temperature: float,
                            prev_a_list: list[str], prev_b_list: list[str],
                            used_words_list: list[set[str]],
                            used_stems_list: list[set[str]],
                            stem_fn,
                            bridge_threshold: float,
                            cot_tokens: int = 80,
                            adapter_name: "str | None" = None,
                            ) -> list[tuple[str, str, float, list[tuple[str, str, str]], bool]]:
    """
    Batched version of get_valid_word. Finds a valid word for each of N independent
    contexts; retries (up to MAX_RETRIES rounds) only the episodes that haven't found
    a valid word yet, batching generation across all still-pending episodes each round.

    Returns a list of (word, prompt_text, penalty, all_attempts, exhausted), one per
    entry in messages_list, in the same order.
    """
    n = len(messages_list)
    results: list[Optional[tuple]] = [None] * n
    attempts_per_ep: list[list[tuple[str, str, str]]] = [[] for _ in range(n)]
    penalties = [0.0] * n
    pending = list(range(n))

    for attempt_idx in range(MAX_RETRIES):
        if not pending:
            break

        batch_messages = [messages_list[i] for i in pending]
        gen_results = _generate_cot_batch(
            model, tokenizer, batch_messages, device, temperature,
            max_new_tokens=cot_tokens, adapter_name=adapter_name,
        )

        still_pending = []
        for k, i in enumerate(pending):
            prompt_text, raw_full = gen_results[k]
            cot_prefix, word_str = parse_cot_output(raw_full)
            train_prompt = prompt_text + cot_prefix
            attempts_per_ep[i].append((train_prompt, word_str, raw_full))

            word = sanitize_output(word_str)
            valid = (
                word is not None
                and is_in_wordnet(word)
                and word not in used_words_list[i]
                and stem_fn(word) not in used_stems_list[i]
                and is_valid_bridge(word, prev_a_list[i], prev_b_list[i], bridge_threshold)
            )

            if valid:
                results[i] = (word, prompt_text, penalties[i], attempts_per_ep[i], False)
            else:
                penalties[i] += RETRY_PENALTY
                still_pending.append(i)

        pending = still_pending

    for i in pending:
        last_train_prompt, last_word_str, _ = attempts_per_ep[i][-1]
        last_word = sanitize_output(last_word_str) or "unknown"
        results[i] = (last_word, last_train_prompt, penalties[i], attempts_per_ep[i], True)

    return results


def _generate_batch_fast(model, tokenizer, messages_list: list[list[dict]],
                         device: str, temperature: float = 0.9,
                         max_new_tokens: int = 8,
                         adapter_name: "str | None" = None) -> list[tuple[str, str]]:
    """Batch-generate short word-only completions (no CoT) for training rollouts."""
    if not messages_list:
        return []
    if adapter_name is not None:
        model.set_adapter(adapter_name)

    prompts = []
    for messages in messages_list:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(m["content"])
                elif m["role"] == "user":
                    parts.append(f"\nUser: {m['content']}\nAssistant:")
            prompt_text = "\n".join(parts)
        prompts.append(prompt_text)

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=768,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-3),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                *tokenizer.encode("\n", add_special_tokens=False),
            ],
        )

    prompt_len = enc["input_ids"].shape[1]
    return [
        (prompts[i], tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip())
        for i in range(len(prompts))
    ]


def get_valid_words_batched_fast(model, tokenizer, messages_list: list[list[dict]],
                                 device: str, temperature: float,
                                 prev_a_list: list[str], prev_b_list: list[str],
                                 used_words_list: list[set[str]],
                                 used_stems_list: list[set[str]],
                                 stem_fn,
                                 bridge_threshold: float,
                                 adapter_name: "str | None" = None,
                                 ) -> list[tuple[str, str, float, list[tuple[str, str, str]], bool]]:
    """
    Fast (non-CoT) counterpart of get_valid_words_batched, used for the actual
    training rollouts. Same retry/validity logic, no reasoning prefix, and
    attempts entries are (prompt_text, word_str, word_str) so the tuple shape
    stays compatible with plays_to_rollout and the shared logging helpers.
    """
    n = len(messages_list)
    results: list[Optional[tuple]] = [None] * n
    attempts_per_ep: list[list[tuple[str, str, str]]] = [[] for _ in range(n)]
    penalties = [0.0] * n
    pending = list(range(n))

    for attempt_idx in range(MAX_RETRIES):
        if not pending:
            break

        batch_messages = [messages_list[i] for i in pending]
        gen_results = _generate_batch_fast(
            model, tokenizer, batch_messages, device, temperature,
            adapter_name=adapter_name,
        )

        still_pending = []
        for k, i in enumerate(pending):
            prompt_text, raw = gen_results[k]
            attempts_per_ep[i].append((prompt_text, raw, raw))

            word = sanitize_output(raw)
            valid = (
                word is not None
                and is_in_wordnet(word)
                and word not in used_words_list[i]
                and stem_fn(word) not in used_stems_list[i]
                and is_valid_bridge(word, prev_a_list[i], prev_b_list[i], bridge_threshold)
            )

            if valid:
                results[i] = (word, prompt_text, penalties[i], attempts_per_ep[i], False)
            else:
                penalties[i] += RETRY_PENALTY
                still_pending.append(i)

        pending = still_pending

    for i in pending:
        last_prompt, last_word_str, _ = attempts_per_ep[i][-1]
        last_word = sanitize_output(last_word_str) or "unknown"
        results[i] = (last_word, last_prompt, penalties[i], attempts_per_ep[i], True)

    return results


# ---------------------------------------------------------------------------
# 6.  Reward functions
# ---------------------------------------------------------------------------

def turn_reward(word: str, anchor_a: str, anchor_b: str,
                retry_penalty: float, threshold: float) -> float:
    r = 0.0
    r -= retry_penalty

    if not is_in_wordnet(word):
        r -= 1.5
        return r

    sim_a = _best_sim(word, anchor_a)
    sim_b = _best_sim(word, anchor_b)
    bridge_score = sim_a + sim_b
    r += bridge_score * 2.0

    if sim_a < threshold or sim_b < threshold:
        r -= 3.0

    return r


def episode_terminal_reward(won: bool, n_rounds: int, max_rounds: int) -> float:
    if won:
        efficiency = (max_rounds - n_rounds) / max(max_rounds - 1, 1)
        return 5.0 + 15.0 * efficiency
    else:
        return -5.0


# ---------------------------------------------------------------------------
# 7.  Episode runner
# ---------------------------------------------------------------------------

@dataclass
class AgentTurn:
    attempts:       list[tuple[str, str, str]]  # (train_prompt, word_str, raw_full)
    word:           str
    is_valid_bridge: bool
    turn_reward:    float


@dataclass
class Episode:
    start_a:   str
    start_b:   str
    turns_a:   list[AgentTurn] = field(default_factory=list)
    turns_b:   list[AgentTurn] = field(default_factory=list)
    won:       bool = False
    n_rounds:  int  = 0
    exhausted: bool = False


def run_episode(model_a, model_b, tokenizer_a, tokenizer_b,
                start_a: str, start_b: str,
                device: str, max_rounds: int = 10,
                temperature: float = 0.9,
                bridge_threshold: float = 0.10,
                cot_tokens: int = 80,
                adapter_name_a: "str | None" = None,
                adapter_name_b: "str | None" = None) -> Episode:
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        def _stem(w): return stemmer.stem(w)
    except Exception:
        def _stem(w): return w

    ep = Episode(start_a=start_a, start_b=start_b)

    prev_a = start_a
    prev_b = start_b
    used_words: set[str] = {start_a, start_b}
    used_stems: set[str] = {_stem(start_a), _stem(start_b)}
    history: list[dict] = []

    for round_idx in range(max_rounds):
        msgs_a = build_messages(prev_a, prev_b, history, "A", start_a, start_b)
        word_a, _, penalty_a, attempts_a, exhausted_a = get_valid_word(
            model_a, tokenizer_a, msgs_a, device, temperature,
            prev_a, prev_b, used_words, used_stems, _stem, bridge_threshold,
            cot_tokens=cot_tokens, adapter_name=adapter_name_a,
        )

        msgs_b = build_messages(prev_a, prev_b, history, "B", start_a, start_b)
        word_b, _, penalty_b, attempts_b, exhausted_b = get_valid_word(
            model_b, tokenizer_b, msgs_b, device, temperature,
            prev_a, prev_b, used_words, used_stems, _stem, bridge_threshold,
            cot_tokens=cot_tokens, adapter_name=adapter_name_b,
        )

        r_a = turn_reward(word_a, prev_a, prev_b, penalty_a, bridge_threshold)
        r_b = turn_reward(word_b, prev_a, prev_b, penalty_b, bridge_threshold)

        valid_a = is_valid_bridge(word_a, prev_a, prev_b, bridge_threshold)
        valid_b = is_valid_bridge(word_b, prev_a, prev_b, bridge_threshold)

        ep.turns_a.append(AgentTurn(attempts_a, word_a, valid_a, r_a))
        ep.turns_b.append(AgentTurn(attempts_b, word_b, valid_b, r_b))

        ep.n_rounds = round_idx + 1

        if exhausted_a or exhausted_b:
            ep.exhausted = True
            break

        if word_a == word_b:
            ep.won = True
            break

        prev_a = word_a
        prev_b = word_b
        used_words.add(word_a);  used_stems.add(_stem(word_a))
        used_words.add(word_b);  used_stems.add(_stem(word_b))
        history.append({"word_a": word_a, "word_b": word_b})

    terminal = episode_terminal_reward(ep.won, ep.n_rounds, max_rounds)
    if ep.exhausted:
        terminal -= 5.0

    if ep.turns_a:
        ep.turns_a[-1].turn_reward += terminal
    if ep.turns_b:
        ep.turns_b[-1].turn_reward += terminal

    return ep


def run_episodes_batched(model_a, model_b, tokenizer_a, tokenizer_b,
                         start_a: str, start_b: str, n_plays: int,
                         device: str, max_rounds: int = 10,
                         temperature: float = 0.9,
                         bridge_threshold: float = 0.10,
                         cot_tokens: int = 80,
                         adapter_name_a: "str | None" = None,
                         adapter_name_b: "str | None" = None) -> list[Episode]:
    """
    Run n_plays independent episodes sharing the same starting pair, batching
    generation across all still-active episodes each round (mirrors
    run_mind_episodes_batch in grpo_mind_modified.py). Each play's dialogue
    history diverges independently after round 1 due to sampling.
    """
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        def _stem(w): return stemmer.stem(w)
    except Exception:
        def _stem(w): return w

    eps        = [Episode(start_a=start_a, start_b=start_b) for _ in range(n_plays)]
    prev_a     = [start_a] * n_plays
    prev_b     = [start_b] * n_plays
    used_words = [{start_a, start_b} for _ in range(n_plays)]
    used_stems = [{_stem(start_a), _stem(start_b)} for _ in range(n_plays)]
    history: list[list[dict]] = [[] for _ in range(n_plays)]
    done       = [False] * n_plays

    for round_idx in range(max_rounds):
        alive = [i for i in range(n_plays) if not done[i]]
        if not alive:
            break

        msgs_a = [build_messages(prev_a[i], prev_b[i], history[i], "A", start_a, start_b)
                  for i in alive]
        results_a = get_valid_words_batched(
            model_a, tokenizer_a, msgs_a, device, temperature,
            [prev_a[i] for i in alive], [prev_b[i] for i in alive],
            [used_words[i] for i in alive], [used_stems[i] for i in alive],
            _stem, bridge_threshold, cot_tokens=cot_tokens, adapter_name=adapter_name_a,
        )

        msgs_b = [build_messages(prev_a[i], prev_b[i], history[i], "B", start_a, start_b)
                  for i in alive]
        results_b = get_valid_words_batched(
            model_b, tokenizer_b, msgs_b, device, temperature,
            [prev_a[i] for i in alive], [prev_b[i] for i in alive],
            [used_words[i] for i in alive], [used_stems[i] for i in alive],
            _stem, bridge_threshold, cot_tokens=cot_tokens, adapter_name=adapter_name_b,
        )

        for k, i in enumerate(alive):
            word_a, _, penalty_a, attempts_a, exhausted_a = results_a[k]
            word_b, _, penalty_b, attempts_b, exhausted_b = results_b[k]

            r_a = turn_reward(word_a, prev_a[i], prev_b[i], penalty_a, bridge_threshold)
            r_b = turn_reward(word_b, prev_a[i], prev_b[i], penalty_b, bridge_threshold)

            valid_a = is_valid_bridge(word_a, prev_a[i], prev_b[i], bridge_threshold)
            valid_b = is_valid_bridge(word_b, prev_a[i], prev_b[i], bridge_threshold)

            ep = eps[i]
            ep.turns_a.append(AgentTurn(attempts_a, word_a, valid_a, r_a))
            ep.turns_b.append(AgentTurn(attempts_b, word_b, valid_b, r_b))
            ep.n_rounds = round_idx + 1

            if exhausted_a or exhausted_b:
                ep.exhausted = True
                done[i] = True
                continue

            if word_a == word_b:
                ep.won = True
                done[i] = True
                continue

            prev_a[i] = word_a
            prev_b[i] = word_b
            used_words[i].add(word_a); used_stems[i].add(_stem(word_a))
            used_words[i].add(word_b); used_stems[i].add(_stem(word_b))
            history[i].append({"word_a": word_a, "word_b": word_b})

    for ep in eps:
        terminal = episode_terminal_reward(ep.won, ep.n_rounds, max_rounds)
        if ep.exhausted:
            terminal -= 5.0
        if ep.turns_a:
            ep.turns_a[-1].turn_reward += terminal
        if ep.turns_b:
            ep.turns_b[-1].turn_reward += terminal

    return eps


def run_episodes_batched_fast(model_a, model_b, tokenizer_a, tokenizer_b,
                              start_a: str, start_b: str, n_plays: int,
                              device: str, max_rounds: int = 10,
                              temperature: float = 0.9,
                              bridge_threshold: float = 0.10,
                              adapter_name_a: "str | None" = None,
                              adapter_name_b: "str | None" = None) -> list[Episode]:
    """
    Fast (non-CoT) counterpart of run_episodes_batched — this is what actually
    generates the training rollouts every step. Identical batching/retry
    structure, word-only generation (build_messages_fast, 8 tokens instead of
    cot_tokens). CoT rollouts are reserved for the periodic side-eval driven
    by --cot_eval_every in train().
    """
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        def _stem(w): return stemmer.stem(w)
    except Exception:
        def _stem(w): return w

    eps        = [Episode(start_a=start_a, start_b=start_b) for _ in range(n_plays)]
    prev_a     = [start_a] * n_plays
    prev_b     = [start_b] * n_plays
    used_words = [{start_a, start_b} for _ in range(n_plays)]
    used_stems = [{_stem(start_a), _stem(start_b)} for _ in range(n_plays)]
    history: list[list[dict]] = [[] for _ in range(n_plays)]
    done       = [False] * n_plays

    for round_idx in range(max_rounds):
        alive = [i for i in range(n_plays) if not done[i]]
        if not alive:
            break

        msgs_a = [build_messages_fast(prev_a[i], prev_b[i], history[i], "A", start_a, start_b)
                  for i in alive]
        results_a = get_valid_words_batched_fast(
            model_a, tokenizer_a, msgs_a, device, temperature,
            [prev_a[i] for i in alive], [prev_b[i] for i in alive],
            [used_words[i] for i in alive], [used_stems[i] for i in alive],
            _stem, bridge_threshold, adapter_name=adapter_name_a,
        )

        msgs_b = [build_messages_fast(prev_a[i], prev_b[i], history[i], "B", start_a, start_b)
                  for i in alive]
        results_b = get_valid_words_batched_fast(
            model_b, tokenizer_b, msgs_b, device, temperature,
            [prev_a[i] for i in alive], [prev_b[i] for i in alive],
            [used_words[i] for i in alive], [used_stems[i] for i in alive],
            _stem, bridge_threshold, adapter_name=adapter_name_b,
        )

        for k, i in enumerate(alive):
            word_a, _, penalty_a, attempts_a, exhausted_a = results_a[k]
            word_b, _, penalty_b, attempts_b, exhausted_b = results_b[k]

            r_a = turn_reward(word_a, prev_a[i], prev_b[i], penalty_a, bridge_threshold)
            r_b = turn_reward(word_b, prev_a[i], prev_b[i], penalty_b, bridge_threshold)

            valid_a = is_valid_bridge(word_a, prev_a[i], prev_b[i], bridge_threshold)
            valid_b = is_valid_bridge(word_b, prev_a[i], prev_b[i], bridge_threshold)

            ep = eps[i]
            ep.turns_a.append(AgentTurn(attempts_a, word_a, valid_a, r_a))
            ep.turns_b.append(AgentTurn(attempts_b, word_b, valid_b, r_b))
            ep.n_rounds = round_idx + 1

            if exhausted_a or exhausted_b:
                ep.exhausted = True
                done[i] = True
                continue

            if word_a == word_b:
                ep.won = True
                done[i] = True
                continue

            prev_a[i] = word_a
            prev_b[i] = word_b
            used_words[i].add(word_a); used_stems[i].add(_stem(word_a))
            used_words[i].add(word_b); used_stems[i].add(_stem(word_b))
            history[i].append({"word_a": word_a, "word_b": word_b})

    for ep in eps:
        terminal = episode_terminal_reward(ep.won, ep.n_rounds, max_rounds)
        if ep.exhausted:
            terminal -= 5.0
        if ep.turns_a:
            ep.turns_a[-1].turn_reward += terminal
        if ep.turns_b:
            ep.turns_b[-1].turn_reward += terminal

    return eps


# ---------------------------------------------------------------------------
# 8.  Plays → Rollout conversion
# ---------------------------------------------------------------------------

def plays_to_rollout(plays: list[Episode], agent: str, group_id: int) -> Rollout:
    """
    Convert G independent plays of the same starting pair into one Rollout.

    Reward normalization is done at the PLAY level before flattening:
    - Compute one scalar reward per play (sum of all turn rewards, which
      includes the terminal reward on the last turn).
    - Z-score normalize across the G play rewards → advantages.
    - Assign each play's advantage to all its (prompt, completion) pairs.

    This avoids a bias in grpo_loss's within-group normalization, which would
    otherwise weight longer plays (more turns → more samples) more heavily.

    Only the accepted attempt (last entry in turn.attempts) is included per
    turn — retry attempts are excluded because they don't deserve credit or
    blame for the game outcome.

    For retry-exhausted games, the final turn's completion is the last word
    the model tried before giving up — a rejected word, not a real game
    action. That turn is excluded from the rollout. The terminal penalty for
    the exhausted game still flows into the play's z-scored advantage (since
    play_rewards sums ALL turn rewards including the exhausted turn), so the
    negative signal reaches the earlier valid turns via their shared advantage.
    """
    play_rewards = []
    for ep in plays:
        turns = ep.turns_a if agent == "a" else ep.turns_b
        play_rewards.append(sum(t.turn_reward for t in turns))

    r = torch.tensor(play_rewards, dtype=torch.float32)
    if r.std() > 1e-6:
        advantages = ((r - r.mean()) / (r.std() + 1e-8)).tolist()
    else:
        advantages = [0.0] * len(plays)

    prompts, completions, rewards = [], [], []
    for ep, adv in zip(plays, advantages):
        turns = ep.turns_a if agent == "a" else ep.turns_b
        # Skip the final turn of exhausted games: its completion is the last
        # rejected word from the retry loop, not a real game action.
        turns_to_include = turns[:-1] if ep.exhausted else turns
        for turn in turns_to_include:
            # train_prompt = original prompt + CoT prefix; completion = word only
            train_prompt, word_str, _ = turn.attempts[-1]
            prompts.append(train_prompt)
            completions.append(word_str)
            rewards.append(adv)

    return Rollout(prompts=prompts, completions=completions,
                   rewards=rewards, group_id=group_id)


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------

def _make_run_dir(args) -> str:
    """
    Create a unique subdirectory under args.output_dir for this run's logs.

    Format: {output_dir}/{model_short}_lr{lr}_kl{kl}_ep{episodes}_YYYYMMDD-HHMMSS

    Only logs and the loss curve go here. Checkpoints overwrite a single
    model_a_latest/ and model_b_latest/ inside this dir to save disk space.
    """
    def short(name):
        return name.split("/")[-1]

    model_tag = (
        short(args.model_a) if args.model_a == args.model_b
        else f"{short(args.model_a)}_vs_{short(args.model_b)}"
    )
    ts = datetime.now().strftime("%Y_%m_%d__%H_%M")
    run_name = f"{model_tag}_lr{args.learning_rate}_kl{args.kl_coef}_B{args.num_groups}G{args.plays_per_group}_cot{args.cot_tokens}_coteval{args.cot_eval_every}_steps{args.total_steps}_{ts}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    return run_dir, run_name


def train(args):
    if not WORDNET_AVAILABLE:
        raise ImportError(
            "pip install nltk && python -c \"import nltk; "
            "nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 on A40 (compute capability 8.6) — bfloat16 is only preferable on A100+
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    concrete_nouns = get_concrete_nouns()

    # --- Load models ---------------------------------------------------------
    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_a.eos_token is None:
        tokenizer_a.eos_token = tokenizer_a.pad_token
    tokenizer_a.padding_side = "left"

    if args.model_b == args.model_a:
        # Dual-adapter: one base model, two named LoRA adapters, one shared ref.
        # Memory: 1×base (16GB) + 1×ref (16GB) + 2×adapters (~200MB) ≈ 32GB total.
        # Fits a 44GB A40 with ~12GB headroom for activations.
        print(f"Loading dual-adapter policy: {args.model_a}")
        shared_policy = make_dual_adapter_policy(
            args.model_a, args.lora_r, args.lora_alpha, dtype,
        )
        shared_ref  = make_ref(args.model_a, dtype)
        model_a = model_b = shared_policy
        ref_a   = ref_b   = shared_ref
        tokenizer_b  = tokenizer_a
        adapter_name_a, adapter_name_b = "agent_a", "agent_b"
    else:
        print(f"Loading model A: {args.model_a}")
        model_a = make_policy(args.model_a, args.use_peft, args.lora_r, args.lora_alpha, dtype)
        ref_a   = make_ref(args.model_a, dtype)
        print(f"Loading model B: {args.model_b}")
        model_b = make_policy(args.model_b, args.use_peft, args.lora_r, args.lora_alpha, dtype)
        ref_b   = make_ref(args.model_b, dtype)
        tokenizer_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
        if tokenizer_b.pad_token is None:
            tokenizer_b.pad_token = tokenizer_b.eos_token
        if tokenizer_b.eos_token is None:
            tokenizer_b.eos_token = tokenizer_b.pad_token
        tokenizer_b.padding_side = "left"
        adapter_name_a = adapter_name_b = None

    # --- GRPOLoop for each agent (8-bit Adam, gradient accumulation) --------
    loop_a = GRPOLoop(
        model=model_a, ref_model=ref_a, tokenizer=tokenizer_a, device=device,
        learning_rate=args.learning_rate, kl_coef=args.kl_coef,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_name=adapter_name_a,
        precomputed_advantages=True,
    )
    loop_b = GRPOLoop(
        model=model_b, ref_model=ref_b, tokenizer=tokenizer_b, device=device,
        learning_rate=args.learning_rate, kl_coef=args.kl_coef,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_name=adapter_name_b,
        precomputed_advantages=True,
    )

    # --- Output paths --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir, run_name = _make_run_dir(args)
    game_log_path     = os.path.join(run_dir, "game_log.txt")
    words_log_path    = os.path.join(run_dir, "game_log_words.txt")
    cot_eval_log_path = os.path.join(run_dir, "cot_eval_log.txt")
    loss_plot_path     = os.path.join(run_dir, "loss_curve.png")
    cot_eval_plot_path = os.path.join(run_dir, "cot_eval_curve.png")
    loss_history: list[dict] = []
    cot_eval_history: list[dict] = []

    # --- Write run header to both logs ---------------------------------------
    def _write_header(f):
        f.write(f"\n{'#'*70}\n")
        f.write(f"# NEW RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# model_a={args.model_a}\n")
        f.write(f"# model_b={args.model_b}\n")
        f.write(f"# lr={args.learning_rate}  kl_coef={args.kl_coef}  "
                f"num_groups={args.num_groups}  plays_per_group={args.plays_per_group}\n")
        f.write(f"# max_rounds={args.max_rounds}  bridge_threshold={args.bridge_threshold}  "
                f"temperature={args.temperature}  cot_tokens={args.cot_tokens}\n")
        f.write(f"# cot_eval_every={args.cot_eval_every}  cot_eval_groups={args.cot_eval_groups}  "
                f"cot_eval_plays={args.cot_eval_plays}\n")
        f.write(f"# total_steps={args.total_steps}  grad_accum={args.gradient_accumulation_steps}\n")
        f.write(f"# use_peft={args.use_peft}  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}\n")
        f.write(f"# run_dir={run_dir}\n")
        f.write(f"{'#'*70}\n")

    with open(game_log_path, "a") as f:
        _write_header(f)
    with open(words_log_path, "a") as f:
        _write_header(f)
    with open(cot_eval_log_path, "a") as f:
        _write_header(f)

    global_step = 0

    for batch_idx in range(args.total_steps):

        # Sample B sufficiently-dissimilar starting pairs
        starts = []
        while len(starts) < args.num_groups:
            a, b = random.sample(concrete_nouns, 2)
            if _best_path_sim(a, b) < 0.5:
                starts.append((a, b))

        # --- Rollout: G independent plays per starting pair, fast (non-CoT)
        # path. This is what training actually trains on every step; batched
        # across all G plays each round, mirroring run_mind_episodes_batch. --
        model_a.eval(); model_b.eval()
        all_plays = []   # [group_idx][play_idx] = Episode
        for (sa, sb) in starts:
            group_plays = run_episodes_batched(
                model_a, model_b, tokenizer_a, tokenizer_b,
                sa, sb, args.plays_per_group, device,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                bridge_threshold=args.bridge_threshold,
                cot_tokens=args.cot_tokens,
                adapter_name_a=adapter_name_a,
                adapter_name_b=adapter_name_b,
            )
            all_plays.append(group_plays)

        # --- Stats across all plays ------------------------------------------
        flat = [ep for group in all_plays for ep in group]
        wins       = sum(ep.won for ep in flat)
        avg_rounds = sum(ep.n_rounds for ep in flat) / len(flat)
        exhausted  = sum(ep.exhausted for ep in flat)
        win_rate   = wins / len(flat)
        print(f"  step={global_step:4d} | "
              f"wins={wins}/{len(flat)} | "
              f"avg_rounds={avg_rounds:.1f} | "
              f"retry_exhausted={exhausted}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ep in flat:
            _log_episode_to_file(ep, global_step, game_log_path, timestamp)
            _log_episode_words_to_file(ep, global_step, words_log_path, timestamp)

        # --- Periodic CoT side-eval: generates with the "Thinking:" preamble
        # to track how reasoning-conditioned play evolves, but never feeds
        # gradient and never touches the rollouts used for training above. --
        if args.cot_eval_every > 0 and global_step % args.cot_eval_every == 0:
            cot_starts = []
            while len(cot_starts) < args.cot_eval_groups:
                a, b = random.sample(concrete_nouns, 2)
                if _best_path_sim(a, b) < 0.5:
                    cot_starts.append((a, b))

            cot_plays = []
            for (sa, sb) in cot_starts:
                group_plays = run_episodes_batched(
                    model_a, model_b, tokenizer_a, tokenizer_b,
                    sa, sb, args.cot_eval_plays, device,
                    max_rounds=args.max_rounds,
                    temperature=args.temperature,
                    bridge_threshold=args.bridge_threshold,
                    cot_tokens=args.cot_tokens,
                    adapter_name_a=adapter_name_a,
                    adapter_name_b=adapter_name_b,
                )
                cot_plays.append(group_plays)

            flat_cot       = [ep for group in cot_plays for ep in group]
            wins_cot       = sum(ep.won for ep in flat_cot)
            avg_rounds_cot = sum(ep.n_rounds for ep in flat_cot) / len(flat_cot)
            win_rate_cot   = wins_cot / len(flat_cot)
            print(f"  [cot-eval] step={global_step:4d} | "
                  f"wins={wins_cot}/{len(flat_cot)} | "
                  f"avg_rounds={avg_rounds_cot:.1f}")

            for ep in flat_cot:
                _log_episode_to_file(ep, global_step, cot_eval_log_path, timestamp)

            cot_eval_history.append({
                "step":       global_step,
                "win_rate":   win_rate_cot,
                "avg_rounds": avg_rounds_cot,
            })
            try:
                _save_cot_eval_curve(cot_eval_history, cot_eval_plot_path)
            except Exception as e:
                print(f"  [plot] WARNING: failed to save cot eval curve: {e}")

        if args.eval_only:
            for ep in flat[:2]:
                _print_episode(ep)
            global_step += 1
            continue

        # --- Build rollouts: one per group, G plays each ---------------------
        rollouts_a = [plays_to_rollout(group, "a", gid)
                      for gid, group in enumerate(all_plays)]
        rollouts_b = [plays_to_rollout(group, "b", gid)
                      for gid, group in enumerate(all_plays)]

        # Process one group at a time so each backward pass sees G plays
        # (same memory footprint as a single-group run). Gradients accumulate
        # across num_groups calls before the optimizer fires.
        loss_a_val = None
        for r in rollouts_a:
            val = loop_a.step([r])
            if val is not None:
                loss_a_val = val

        loss_b_val = None
        for r in rollouts_b:
            val = loop_b.step([r])
            if val is not None:
                loss_b_val = val

        # Only log when optimizer actually stepped (every grad_accum calls)
        if loss_a_val is not None and loss_b_val is not None:
            print(f"           loss_a={loss_a_val:.4f} | "
                  f"loss_b={loss_b_val:.4f}")
            loss_history.append({
                "step":       global_step,
                "loss_a":     loss_a_val,
                "loss_b":     loss_b_val,
                "win_rate":   win_rate,
                "avg_rounds": avg_rounds,
            })
            try:
                _save_loss_curve(loss_history, loss_plot_path)
            except Exception as e:
                print(f"  [plot] WARNING: failed to save loss curve: {e}")

        # --- Checkpoint (overwrites latest — saves disk space) ---------------
        if global_step % args.save_steps == 0 and global_step > 0:
            loop_a.save(run_dir, "a_latest")
            loop_b.save(run_dir, "b_latest")

        global_step += 1

    if not args.eval_only:
        loop_a.save(run_dir, "a_final")
        loop_b.save(run_dir, "b_final")


# ---------------------------------------------------------------------------
# 10. Evaluation helpers
# ---------------------------------------------------------------------------

def _log_episode_to_file(ep: Episode, global_step: int, log_path: str,
                          timestamp: str):
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{timestamp}] step={global_step}  "
                f"'{ep.start_a}' ↔ '{ep.start_b}'\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Round':<8} {'Agent A':<24} {'Agent B':<24}\n")
        f.write(f"  {'-'*56}\n")
        for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
            bridge_a  = "✓" if ta.is_valid_bridge else "✗"
            bridge_b  = "✓" if tb.is_valid_bridge else "✗"
            retries_a = len(ta.attempts) - 1
            retries_b = len(tb.attempts) - 1
            retry_a   = f"[r{retries_a}]" if retries_a > 0 else ""
            retry_b   = f"[r{retries_b}]" if retries_b > 0 else ""
            col_a = f"{ta.word}{retry_a}({bridge_a})"
            col_b = f"{tb.word}{retry_b}({bridge_b})"
            f.write(f"  {i+1:<8} {col_a:<24} {col_b:<24}\n")
            # Show CoT from the accepted attempt
            if ta.attempts:
                _, _, raw_a = ta.attempts[-1]
                f.write(f"    CoT_A: {raw_a}\n")
            if tb.attempts:
                _, _, raw_b = tb.attempts[-1]
                f.write(f"    CoT_B: {raw_b}\n")
        outcome = ("WON" if ep.won
                   else ("EXHAUSTED" if ep.exhausted else "TIMEOUT"))
        f.write(f"  → {outcome} in {ep.n_rounds} round(s)\n")

def _log_episode_words_to_file(ep: Episode, global_step: int, log_path: str,
                               timestamp: str):
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{timestamp}] step={global_step}  "
                f"'{ep.start_a}' ↔ '{ep.start_b}'\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Round':<8} {'Agent A':<24} {'Agent B':<24}\n")
        f.write(f"  {'-'*56}\n")
        for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
            bridge_a  = "✓" if ta.is_valid_bridge else "✗"
            bridge_b  = "✓" if tb.is_valid_bridge else "✗"
            retries_a = len(ta.attempts) - 1
            retries_b = len(tb.attempts) - 1
            retry_a   = f"[r{retries_a}]" if retries_a > 0 else ""
            retry_b   = f"[r{retries_b}]" if retries_b > 0 else ""
            col_a = f"{ta.word}{retry_a}({bridge_a})"
            col_b = f"{tb.word}{retry_b}({bridge_b})"
            f.write(f"  {i+1:<8} {col_a:<24} {col_b:<24}\n")
        outcome = ("WON" if ep.won
                   else ("EXHAUSTED" if ep.exhausted else "TIMEOUT"))
        f.write(f"  → {outcome} in {ep.n_rounds} round(s)\n")

def _save_loss_curve(loss_history: list[dict], plot_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not found — skipping loss curve.")
        return

    steps  = [r["step"]   for r in loss_history]
    loss_a = [r["loss_a"] for r in loss_history]
    loss_b = [r["loss_b"] for r in loss_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: GRPO loss
    ax1.plot(steps, loss_a, label="loss_a", color="steelblue",  linewidth=1.2)
    ax1.plot(steps, loss_b, label="loss_b", color="darkorange", linewidth=1.2)
    ax1.set_ylabel("GRPO loss")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Bottom panel: win rate (left) + avg rounds to converge (right)
    has_win_rate   = "win_rate"   in loss_history[0]
    has_avg_rounds = "avg_rounds" in loss_history[0]

    if has_win_rate:
        win_rates = [r["win_rate"] for r in loss_history]
        ax2.plot(steps, win_rates, label="win_rate", color="green",
                 linewidth=1.2)
        ax2.set_ylabel("win rate")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)

    if has_avg_rounds:
        avg_rounds_vals = [r["avg_rounds"] for r in loss_history]
        ax2r = ax2.twinx()
        ax2r.plot(steps, avg_rounds_vals, label="avg rounds", color="purple",
                  linewidth=1.2, linestyle="--", alpha=0.85)
        ax2r.set_ylabel("avg rounds to converge")
        ax2r.legend(loc="upper right")

    ax2.set_xlabel("step")

    fig.suptitle("GRPO Converge CoT — training curves (short CoT rollouts)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Loss curve saved → {plot_path}")


def _save_cot_eval_curve(cot_eval_history: list[dict], plot_path: str):
    """
    Plots the periodic CoT side-eval (win rate / avg rounds when the model
    reasons before answering). This tracks how CoT-conditioned play evolves
    over training without ever feeding gradient — see --cot_eval_every.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not found — skipping CoT eval curve.")
        return

    steps      = [r["step"]       for r in cot_eval_history]
    win_rates  = [r["win_rate"]   for r in cot_eval_history]
    avg_rounds = [r["avg_rounds"] for r in cot_eval_history]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, win_rates, label="win_rate (CoT)", color="green",
              marker="o", linewidth=1.2)
    ax1.set_ylabel("win rate")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("step")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, avg_rounds, label="avg rounds (CoT)", color="purple",
              marker="o", linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.set_ylabel("avg rounds to converge")
    ax2.legend(loc="upper right")

    fig.suptitle("GRPO Converge CoT — periodic side-eval (no gradient)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] CoT eval curve saved → {plot_path}")

def _print_episode(ep: Episode):
    print(f"\n  Episode: '{ep.start_a}' ↔ '{ep.start_b}'")
    for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
        bridge_a  = "✓" if ta.is_valid_bridge else "✗"
        bridge_b  = "✓" if tb.is_valid_bridge else "✗"
        retries_a = len(ta.attempts) - 1
        retries_b = len(tb.attempts) - 1
        retry_a   = f"[r{retries_a}]" if retries_a > 0 else ""
        retry_b   = f"[r{retries_b}]" if retries_b > 0 else ""
        print(f"    Round {i+1}: A={ta.word}{retry_a}({bridge_a}) "
              f"B={tb.word}{retry_b}({bridge_b})")
        if ta.attempts:
            _, word_str_a, raw_a = ta.attempts[-1]
            if raw_a != word_str_a:
                print(f"      CoT_A: {raw_a}")
        if tb.attempts:
            _, word_str_b, raw_b = tb.attempts[-1]
            if raw_b != word_str_b:
                print(f"      CoT_B: {raw_b}")
    outcome = "WON" if ep.won else ("EXHAUSTED" if ep.exhausted else "TIMEOUT")
    print(f"  → {outcome} in {ep.n_rounds} round(s)")

def evaluate(model_a, model_b, tokenizer_a, tokenizer_b,
             args, n_episodes: int = 20):
    device = next(model_a.parameters()).device
    model_a.eval(); model_b.eval()

    wins = 0; total_rounds = 0
    concrete_nouns = get_concrete_nouns()
    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    for _ in range(n_episodes):
        sa, sb = random.sample(concrete_nouns, 2)
        ep = run_episode(
            model_a, model_b, tokenizer_a, tokenizer_b,
            sa, sb, str(device),
            max_rounds=args.max_rounds,
            temperature=0.01,
            bridge_threshold=args.bridge_threshold,
            cot_tokens=args.cot_tokens,
        )
        wins += ep.won
        total_rounds += ep.n_rounds
        _print_episode(ep)

    print(f"\nWin rate : {wins}/{n_episodes} = {wins/n_episodes:.1%}")
    print(f"Avg rounds: {total_rounds/n_episodes:.2f}")

# Args

def parse_args():
    p = argparse.ArgumentParser()
    # Game args
    p.add_argument("--model_a",              default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model_b",              default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir",           default="./grpo_converge_cot_output")
    p.add_argument("--max_rounds",           type=int,   default=10)
    p.add_argument("--bridge_threshold",     type=float, default=0.2)
    p.add_argument("--temperature",          type=float, default=0.9)
    p.add_argument("--num_groups",            type=int,   default=4)
    p.add_argument("--plays_per_group",      type=int,   default=8)
    p.add_argument("--eval_only",            action="store_true")
    p.add_argument("--eval_episodes",        type=int,   default=5)
    p.add_argument("--cot_tokens",           type=int,   default=20)
    p.add_argument("--cot_eval_every",       type=int,   default=0,
                   help="Run a CoT side-eval every N steps (0 disables). "
                        "Training rollouts always use the fast, non-CoT path.")
    p.add_argument("--cot_eval_groups",      type=int,   default=1)
    p.add_argument("--cot_eval_plays",       type=int,   default=4)
    # Training args
    p.add_argument("--total_steps",                  type=int,   default=600)
    p.add_argument("--learning_rate",                type=float, default=5e-6)
    p.add_argument("--kl_coef",                      type=float, default=0.04)
    p.add_argument("--gradient_accumulation_steps",  type=int,   default=1)
    p.add_argument("--save_steps",                   type=int,   default=100)
    p.add_argument("--use_peft",             action="store_true")
    p.add_argument("--lora_r",               type=int,   default=16)
    p.add_argument("--lora_alpha",           type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not WORDNET_AVAILABLE:
        print("ERROR: nltk not found. Run:")
        print("  pip install nltk")
        print("  python -c \"import nltk; nltk.download('wordnet'); "
              "nltk.download('omw-1.4')\"")
        exit(1)

    train(args)
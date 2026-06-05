"""
Multi-agent GRPO training for the Converge game — TRL + alternating frozen-agent.

Uses TRL 1.3.0's GRPOTrainer directly. With alternating training, each phase is
single-model training (one model active, opponent frozen), which maps cleanly onto
TRL's single-model API.

Architecture:
  - Two GRPOTrainers (trainer_a, trainer_b), one per model.
  - Each trainer uses a rollout_func that plays the full game: active model (the
    trainer's own model) vs. frozen opponent, returning the active model's FIRST
    TURN completion and the total play reward.
  - TRL normalizes rewards within each group of plays_per_group plays for the same
    starting pair (scale_rewards='group'), equivalent to plays_to_rollout z-scoring.
  - TRL computes ref log-probs via model.disable_adapter() (PEFT base = reference).
    No separate ref model is needed — saves ~16GB VRAM per model.
  - Alternating: trainer.train() for swap_every steps; increment max_steps each
    phase. Optimizer state is preserved between phases.

TRL config mapping:
  per_device_train_batch_size = num_groups       → B starting pairs per optimizer step
  num_generations              = plays_per_group  → G plays per pair
  gradient_accumulation_steps  = 1
  → rollout_func receives B×G prompts; returns B×G completions ordered [pair0×G, pair1×G, ...]
  → TRL's scale_rewards='group' normalises rewards within each group of G (same starting pair)

Usage:
    python grpo_retooled.py --use_peft --total_steps 500 --swap_every 16
    python grpo_retooled.py --eval_only --eval_episodes 5
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
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

try:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1.  WordNet bridge validity
# ---------------------------------------------------------------------------

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
    return _best_sim(word, anchor_a) >= threshold and _best_sim(word, anchor_b) >= threshold

def is_in_wordnet(word: str) -> bool:
    return bool(wn.synsets(word, pos=wn.NOUN))


# ---------------------------------------------------------------------------
# 2.  Concrete noun pool
# ---------------------------------------------------------------------------

CONCRETE_LEXNAMES = {
    "noun.animal", "noun.plant", "noun.food", "noun.artifact",
    "noun.body", "noun.object", "noun.substance",
}
_CONCRETE_NOUN_CACHE: list[str] = []

def get_concrete_nouns(max_words: int = 3000) -> list[str]:
    global _CONCRETE_NOUN_CACHE
    if _CONCRETE_NOUN_CACHE:
        return _CONCRETE_NOUN_CACHE
    try:
        from nltk.corpus import brown
        freq = {}
        for w in brown.words():
            w = w.lower()
            freq[w] = freq.get(w, 0) + 1
        common_words = {w for w, c in freq.items() if c >= 2}
        print(f"[WordNet] Brown corpus loaded: {len(common_words)} common words.")
    except Exception:
        common_words = None
        print("[WordNet] Brown corpus unavailable — no frequency filter applied.")
    seen = set(); words = []
    for synset in wn.all_synsets(pos=wn.NOUN):
        if synset.lexname() not in CONCRETE_LEXNAMES:
            continue
        for lemma in synset.lemmas():
            w = lemma.name().lower().replace("_", "")
            if (w.isalpha() and 3 <= len(w) <= 12
                    and " " not in w and w not in seen
                    and (common_words is None or w in common_words)):
                seen.add(w); words.append(w)
                if len(words) >= max_words:
                    break
        if len(words) >= max_words:
            break
    _CONCRETE_NOUN_CACHE = words
    print(f"[WordNet] Loaded {len(words)} concrete nouns.")
    return words


# ---------------------------------------------------------------------------
# 3.  Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing a word convergence game with a partner.

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

def build_messages(anchor_a: str, anchor_b: str,
                   history: list[dict], agent_label: str,
                   start_a: str = "", start_b: str = "") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not history:
        user_content = (
            f"Game start!\nThe two starting words are: '{anchor_a}' and '{anchor_b}'\n\n"
            f"Round 1: Output ONE word that meaningfully connects BOTH of these words."
        )
    else:
        history_lines = [f"  Round {i+1}: words were '{h['word_a']}' and '{h['word_b']}'"
                         for i, h in enumerate(history)]
        user_content = (
            f"Starting words: '{start_a}' and '{start_b}'\n\n"
            f"History:\n{chr(10).join(history_lines)}\n\n"
            f"Last round: '{anchor_a}' and '{anchor_b}'.\n"
            f"Round {len(history) + 1}: Output ONE word that meaningfully "
            f"connects BOTH '{anchor_a}' and '{anchor_b}'."
        )
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# 4.  Word extraction
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
    for tok in _WORD_RE.findall(text):
        tok = tok.lower()
        if tok not in _PREAMBLE_WORDS and len(tok) >= 2:
            return tok
    return None

def sanitize_output(raw: str) -> Optional[str]:
    return extract_word(raw)


# ---------------------------------------------------------------------------
# 5.  Constrained generation
# ---------------------------------------------------------------------------

RETRY_PENALTY = 1.0
MAX_RETRIES   = 10

def _generate_once(model, tokenizer, messages: list[dict],
                   device: str, temperature: float = 0.9,
                   max_new_tokens: int = 8,
                   adapter_name: "str | None" = None) -> tuple[str, str]:
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(m["content"])
            elif m["role"] == "user":
                parts.append(f"\nUser: {m['content']}\nAssistant:")
        prompt_text = "\n".join(parts)
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=768).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0), temperature=max(temperature, 1e-3),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id,
                          *tokenizer.encode("\n", add_special_tokens=False)],
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, prompt_text


def get_valid_word(model, tokenizer, messages, device, temperature,
                   prev_a, prev_b, used_words, used_stems, stem_fn,
                   bridge_threshold, adapter_name=None):
    all_attempts = []; penalty = 0.0
    for _ in range(MAX_RETRIES):
        raw, prompt_text = _generate_once(model, tokenizer, messages, device,
                                          temperature, adapter_name=adapter_name)
        all_attempts.append((prompt_text, raw))
        word = sanitize_output(raw)
        if word is None: penalty += RETRY_PENALTY; continue
        if not is_in_wordnet(word): penalty += RETRY_PENALTY; continue
        if word in used_words or stem_fn(word) in used_stems:
            penalty += RETRY_PENALTY; continue
        if not is_valid_bridge(word, prev_a, prev_b, bridge_threshold):
            penalty += RETRY_PENALTY; continue
        return word, prompt_text, penalty, all_attempts, False
    last_word = sanitize_output(all_attempts[-1][1]) or "unknown"
    return last_word, all_attempts[-1][0], penalty, all_attempts, True


# ---------------------------------------------------------------------------
# 6.  Reward functions
# ---------------------------------------------------------------------------

def turn_reward(word, anchor_a, anchor_b, retry_penalty, threshold):
    r = -retry_penalty
    if not is_in_wordnet(word):
        return r - 1.5
    sim_a = _best_sim(word, anchor_a); sim_b = _best_sim(word, anchor_b)
    r += (sim_a + sim_b) * 2.0
    if sim_a < threshold or sim_b < threshold:
        r -= 3.0
    return r

def episode_terminal_reward(won, n_rounds, max_rounds):
    if won:
        return 5.0 + 15.0 * (max_rounds - n_rounds) / max(max_rounds - 1, 1)
    return -5.0


# ---------------------------------------------------------------------------
# 7.  Episode runner
# ---------------------------------------------------------------------------

@dataclass
class AgentTurn:
    attempts:        list[tuple[str, str]]
    word:            str
    is_valid_bridge: bool
    turn_reward:     float

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
                start_a, start_b, device, max_rounds=10,
                temperature=0.9, bridge_threshold=0.10,
                adapter_name_a=None, adapter_name_b=None):
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        def _stem(w): return stemmer.stem(w)
    except Exception:
        def _stem(w): return w

    ep = Episode(start_a=start_a, start_b=start_b)
    prev_a, prev_b = start_a, start_b
    used_words = {start_a, start_b}
    used_stems = {_stem(start_a), _stem(start_b)}
    history = []

    for round_idx in range(max_rounds):
        msgs_a = build_messages(prev_a, prev_b, history, "A", start_a, start_b)
        word_a, _, pen_a, att_a, ex_a = get_valid_word(
            model_a, tokenizer_a, msgs_a, device, temperature,
            prev_a, prev_b, used_words, used_stems, _stem, bridge_threshold,
            adapter_name=adapter_name_a)
        msgs_b = build_messages(prev_a, prev_b, history, "B", start_a, start_b)
        word_b, _, pen_b, att_b, ex_b = get_valid_word(
            model_b, tokenizer_b, msgs_b, device, temperature,
            prev_a, prev_b, used_words, used_stems, _stem, bridge_threshold,
            adapter_name=adapter_name_b)

        r_a = turn_reward(word_a, prev_a, prev_b, pen_a, bridge_threshold)
        r_b = turn_reward(word_b, prev_a, prev_b, pen_b, bridge_threshold)
        ep.turns_a.append(AgentTurn(att_a, word_a, is_valid_bridge(word_a, prev_a, prev_b, bridge_threshold), r_a))
        ep.turns_b.append(AgentTurn(att_b, word_b, is_valid_bridge(word_b, prev_a, prev_b, bridge_threshold), r_b))
        ep.n_rounds = round_idx + 1

        if ex_a or ex_b: ep.exhausted = True; break
        if word_a == word_b: ep.won = True; break

        prev_a, prev_b = word_a, word_b
        used_words.update([word_a, word_b])
        used_stems.update([_stem(word_a), _stem(word_b)])
        history.append({"word_a": word_a, "word_b": word_b})

    terminal = episode_terminal_reward(ep.won, ep.n_rounds, max_rounds)
    if ep.exhausted: terminal -= 5.0
    if ep.turns_a: ep.turns_a[-1].turn_reward += terminal
    if ep.turns_b: ep.turns_b[-1].turn_reward += terminal
    return ep


# ---------------------------------------------------------------------------
# 8.  Model loading
# ---------------------------------------------------------------------------

def make_policy(model_name, use_peft, lora_r, lora_alpha, dtype):
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True)
    policy.config.use_cache = False
    if use_peft:
        if not PEFT_AVAILABLE:
            raise ImportError("pip install peft")
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                         target_modules="all-linear",
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        policy = get_peft_model(policy, cfg)
        policy.print_trainable_parameters()
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    return policy

def make_ref(model_name, dtype):
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True)
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    return ref


# ---------------------------------------------------------------------------
# 9.  TRL integration — global context + rollout_func + reward_func
# ---------------------------------------------------------------------------

class _Ctx:
    """Shared state that rollout_func reads at call time."""
    frozen_model    = None
    frozen_tok      = None
    is_a_active     = True
    device          = "cuda"
    args            = None
    nouns           = None
    log_path        = None
    global_step     = 0
    adapter_active  = None   # LoRA adapter name for active model (None if separate models)
    adapter_frozen  = None   # LoRA adapter name for frozen model
    wins_last       = 0
    n_plays_last    = 1
    avg_rounds_last = 0.0

_ctx = _Ctx()


def _game_reward_fn(prompts, completions, game_reward=None, **kwargs):
    """
    TRL reward function: returns pre-computed per-play rewards from rollout_func.
    Passed as extra_field 'game_reward'; TRL normalizes within each group of
    num_generations=plays_per_group completions (same starting pair).
    """
    if game_reward is not None:
        return list(game_reward)
    return [0.0] * len(completions)


def _make_rollout_fn(is_a_active: bool):
    """
    Returns a TRL rollout_func for the given active agent.

    Called by GRPOTrainer._generate() once per mini-batch.
    With per_device_train_batch_size=1 and num_generations=plays_per_group,
    'prompts' has length plays_per_group (one starting pair, G plays).

    Returns the active model's FIRST TURN completion for each play.
    Reward = total play reward (all turn rewards + terminal), passed to
    _game_reward_fn via the 'game_reward' extra field.
    """
    def rollout_fn(prompts, trainer):
        args      = _ctx.args
        active_m  = trainer.model
        active_t  = trainer.processing_class
        frozen_m  = _ctx.frozen_model
        frozen_t  = _ctx.frozen_tok
        device    = _ctx.device

        # Map to model_a / model_b roles
        if is_a_active:
            m_a, m_b, t_a, t_b = active_m, frozen_m, active_t, frozen_t
            adp_a, adp_b = _ctx.adapter_active, _ctx.adapter_frozen
        else:
            m_a, m_b, t_a, t_b = frozen_m, active_m, frozen_t, active_t
            adp_a, adp_b = _ctx.adapter_frozen, _ctx.adapter_active

        # Sample num_groups starting pairs (one per prompt group)
        n_plays  = len(prompts)                          # num_groups * plays_per_group
        plays_pg = args.plays_per_group
        n_groups = max(1, n_plays // plays_pg)

        starts = []
        while len(starts) < n_groups:
            sa, sb = random.sample(_ctx.nouns, 2)
            if _best_path_sim(sa, sb) < 0.5:
                starts.append((sa, sb))

        # Play plays_per_group games per pair; keep order [pair0×G, pair1×G, ...]
        # so TRL's group-level reward normalisation (scale_rewards='group') aligns
        # with starting-pair groups of size plays_per_group.
        episodes = []
        for (sa, sb) in starts:
            for _ in range(plays_pg):
                ep = run_episode(m_a, m_b, t_a, t_b, sa, sb, device,
                                 max_rounds=args.max_rounds,
                                 temperature=args.temperature,
                                 bridge_threshold=args.bridge_threshold,
                                 adapter_name_a=adp_a, adapter_name_b=adp_b)
                episodes.append(ep)

        # Log episodes
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ep in episodes:
            _log_episode_to_file(ep, _ctx.global_step, _ctx.log_path, ts)

        wins      = sum(e.won for e in episodes)
        avg_rnd   = sum(e.n_rounds for e in episodes) / len(episodes)
        exhausted = sum(e.exhausted for e in episodes)
        label     = "A" if is_a_active else "B"
        _ctx.wins_last       = wins
        _ctx.n_plays_last    = len(episodes)
        _ctx.avg_rounds_last = avg_rnd
        print(f"  step={_ctx.global_step:4d} | active={label} | "
              f"wins={wins}/{len(episodes)} | avg_rounds={avg_rnd:.1f} | "
              f"retry_exhausted={exhausted}")
        _ctx.global_step += 1

        # Build TRL output: active model's first-turn completion per play
        pad_id = active_t.pad_token_id or active_t.eos_token_id
        prompt_ids_list  = []
        comp_ids_list    = []
        game_rewards     = []

        for ep in episodes:
            turns = ep.turns_a if is_a_active else ep.turns_b
            if turns:
                prompt_txt, raw_out = turns[0].attempts[-1]
            else:
                prompt_txt, raw_out = "unknown", "unknown"

            total_reward = sum(t.turn_reward for t in turns)
            game_rewards.append(total_reward)

            enc_p = active_t(prompt_txt, return_tensors="pt",
                             truncation=True, max_length=700)
            prompt_ids_list.append(enc_p["input_ids"][0])

            enc_c = active_t(raw_out, return_tensors="pt",
                             add_special_tokens=False, truncation=True, max_length=16)
            c_ids = enc_c["input_ids"][0]
            if c_ids.numel() == 0:
                c_ids = active_t("unknown", return_tensors="pt",
                                 add_special_tokens=False)["input_ids"][0]
            comp_ids_list.append(c_ids)

        # Pad: prompts left-padded, completions right-padded
        max_p = max(t.shape[0] for t in prompt_ids_list)
        max_c = max(t.shape[0] for t in comp_ids_list)
        B = len(episodes)

        prompt_ids = torch.full((B, max_p), pad_id, dtype=torch.long)
        comp_ids   = torch.full((B, max_c), pad_id, dtype=torch.long)
        for i, ids in enumerate(prompt_ids_list):
            prompt_ids[i, max_p - len(ids):] = ids          # left pad
        for i, ids in enumerate(comp_ids_list):
            comp_ids[i, :len(ids)] = ids                    # right pad

        # Per-token logprobs for completions (old logprobs for importance sampling).
        # With TRL's default num_iterations=1 these are used as the reference point
        # (ratio=1 everywhere), but we still compute them correctly.
        prompt_mask = (prompt_ids != pad_id).long()
        comp_mask   = (comp_ids   != pad_id).long()
        full_ids    = torch.cat([prompt_ids, comp_ids], dim=1).to(device)
        full_mask   = torch.cat([prompt_mask, comp_mask], dim=1).to(device)

        with torch.no_grad():
            active_m.eval()
            logits = active_m(input_ids=full_ids, attention_mask=full_mask).logits
        lp = F.log_softmax(logits, dim=-1)   # (B, P+C, V)

        # With left-padded prompts, the last position before the completion
        # (index P-1 in the concatenated tensor) is always the last real prompt token.
        logprobs = torch.zeros(B, max_c, device=device)
        for b in range(B):
            for t in range(max_c):
                if comp_mask[b, t] == 0:
                    break
                tok = comp_ids[b, t].item()
                logprobs[b, t] = lp[b, max_p + t - 1, tok]

        return {
            "prompt_ids":    prompt_ids.to(device),
            "completion_ids": comp_ids.to(device),
            "logprobs":       logprobs,
            "game_reward":    game_rewards,
        }

    return rollout_fn


# ---------------------------------------------------------------------------
# 10.  Metrics callback
# ---------------------------------------------------------------------------

class _LossCallback(TrainerCallback):
    """Captures per-step loss and reward from TRL's log dict."""
    def __init__(self, history: list, active_key: str):
        self.history    = history
        self.active_key = active_key   # "loss_a" or "loss_b"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("train/loss") or logs.get("loss")
        if loss is not None:
            self.history.append({
                "step":          _ctx.global_step,
                self.active_key: loss,
                "reward":        logs.get("train/reward", float("nan")),
                "win_rate":      _ctx.wins_last / max(_ctx.n_plays_last, 1),
                "avg_rounds":    _ctx.avg_rounds_last,
            })


# ---------------------------------------------------------------------------
# 10b. Phase stop callback
# ---------------------------------------------------------------------------

class _PhaseStopCallback(TrainerCallback):
    """Stops training after exactly `limit` optimizer steps — used instead of
    manipulating trainer.args.max_steps, which doesn't survive repeated
    trainer.train() calls reliably."""

    def __init__(self):
        self.limit = 0
        self.count = 0

    def reset(self, limit: int):
        self.limit = limit
        self.count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.count += 1
        if self.count >= self.limit:
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# 11.  Run directory
# ---------------------------------------------------------------------------

def _make_run_dir(args):
    def short(n): return n.split("/")[-1]
    model_tag = (short(args.model_a) if args.model_a == args.model_b
                 else f"{short(args.model_a)}_vs_{short(args.model_b)}")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (f"{model_tag}_lr{args.learning_rate}_kl{args.kl_coef}"
            f"_B{args.num_groups}G{args.plays_per_group}"
            f"_steps{args.total_steps}_alt{args.swap_every}_{ts}")
    d = os.path.join(args.output_dir, name)
    os.makedirs(d, exist_ok=True)
    print(f"Run directory: {d}")
    return d, name


# ---------------------------------------------------------------------------
# 12.  Training loop
# ---------------------------------------------------------------------------

def train(args):
    if not WORDNET_AVAILABLE:
        raise ImportError("pip install nltk; python -c \"import nltk; "
                          "nltk.download('wordnet'); nltk.download('omw-1.4')\"")
    if not TRL_AVAILABLE:
        raise ImportError("pip install trl datasets")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    if args.model_a == args.model_b:
        raise NotImplementedError(
            "Dual-adapter (same base model) not supported in TRL mode. "
            "Use two different model checkpoints.")

    # --- Load models (PEFT wraps base weights = automatic reference model) ---
    print(f"Loading model A: {args.model_a}")
    model_a   = make_policy(args.model_a, args.use_peft, args.lora_r, args.lora_alpha, dtype)
    tok_a     = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token
    tok_a.padding_side = "left"

    print(f"Loading model B: {args.model_b}")
    model_b   = make_policy(args.model_b, args.use_peft, args.lora_r, args.lora_alpha, dtype)
    tok_b     = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    if tok_b.pad_token is None: tok_b.pad_token = tok_b.eos_token
    tok_b.padding_side = "left"

    # --- Output paths --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir, _   = _make_run_dir(args)
    loss_plot    = os.path.join(run_dir, "loss_curve.png")
    loss_history = []

    # --- Global context for rollout_func ------------------------------------
    _ctx.device    = device
    _ctx.args      = args
    _ctx.nouns     = get_concrete_nouns()
    _ctx.log_path  = os.path.join(run_dir, "game_log.txt")
    _ctx.adapter_active = None
    _ctx.adapter_frozen = None

    with open(_ctx.log_path, "a") as f:
        f.write(f"\n{'#'*70}\n"
                f"# NEW RUN (TRL alternating) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# model_a={args.model_a}\n# model_b={args.model_b}\n"
                f"# lr={args.learning_rate}  kl={args.kl_coef}  "
                f"B{args.num_groups}G{args.plays_per_group}  "
                f"swap_every={args.swap_every}  total_steps={args.total_steps}\n"
                f"# run_dir={run_dir}\n{'#'*70}\n")

    # --- Synthetic dataset (content ignored by rollout_func) ----------------
    # Each optimizer step consumes num_groups prompts from the dataset.
    # Make it large enough to never exhaust before the stop callback fires.
    dataset = Dataset.from_dict(
        {"prompt": ["game"] * (args.total_steps * args.num_groups * 4)}
    )

    # --- Shared GRPOConfig --------------------------------------------------
    # per_device_train_batch_size=1:  one starting pair per mini-batch
    # num_generations=plays_per_group: G plays per pair → group reward normalization
    # gradient_accumulation_steps=num_groups: accumulate over all groups before step
    def _make_config() -> GRPOConfig:
        return GRPOConfig(
            output_dir                  = run_dir,
            per_device_train_batch_size = args.num_groups,   # B starting pairs per step
            num_generations             = args.plays_per_group,  # G plays per pair
            gradient_accumulation_steps = 1,
            max_steps                   = args.swap_every * 2,   # callback handles stopping
            learning_rate               = args.learning_rate,
            beta                        = args.kl_coef,
            scale_rewards               = "group",
            fp16                        = (device == "cuda"),
            bf16                        = False,
            gradient_checkpointing      = False,   # already enabled in make_policy
            report_to                   = "none",
            logging_steps               = 1,
            save_strategy               = "no",
            disable_dropout             = True,
            max_completion_length       = 16,      # single word
            temperature                 = args.temperature,
        )

    # --- Create two GRPOTrainers (one per model) ----------------------------
    cb_a   = _LossCallback(loss_history, "loss_a")
    cb_b   = _LossCallback(loss_history, "loss_b")
    stop_a = _PhaseStopCallback()
    stop_b = _PhaseStopCallback()

    trainer_a = GRPOTrainer(
        model            = model_a,
        reward_funcs     = [_game_reward_fn],
        args             = _make_config(),
        train_dataset    = dataset,
        processing_class = tok_a,
        rollout_func     = _make_rollout_fn(is_a_active=True),
        callbacks        = [cb_a, stop_a],
    )
    trainer_b = GRPOTrainer(
        model            = model_b,
        reward_funcs     = [_game_reward_fn],
        args             = _make_config(),
        train_dataset    = dataset,
        processing_class = tok_b,
        rollout_func     = _make_rollout_fn(is_a_active=False),
        callbacks        = [cb_b, stop_b],
    )

    if args.eval_only:
        _run_eval(model_a, model_b, tok_a, tok_b, args, device)
        return

    # --- Alternating training loop ------------------------------------------
    total_done = 0
    phase      = 0

    while total_done < args.total_steps:
        is_a  = (phase % 2 == 0)
        label = "A" if is_a else "B"
        this_phase_steps = min(args.swap_every, args.total_steps - total_done)

        # Set frozen/active context
        _ctx.is_a_active  = is_a
        _ctx.frozen_model = model_b if is_a else model_a
        _ctx.frozen_tok   = tok_b   if is_a else tok_a

        # Freeze opponent, unfreeze active
        active_trainer = trainer_a if is_a else trainer_b
        for p in (model_b if is_a else model_a).parameters():
            p.requires_grad_(False)
        for p in (model_a if is_a else model_b).parameters():
            if hasattr(p, "_is_lora_param") or True:
                pass  # LoRA params managed by PEFT; base weights stay frozen
        # For PEFT models, only LoRA params have requires_grad=True by default.
        # Setting them back:
        if is_a:
            for n, p in model_a.named_parameters():
                if "lora_" in n: p.requires_grad_(True)
        else:
            for n, p in model_b.named_parameters():
                if "lora_" in n: p.requires_grad_(True)

        # Tell the stop callback how many optimizer steps this phase runs
        (stop_a if is_a else stop_b).reset(this_phase_steps)

        print(f"\n{'='*60}")
        print(f"Phase {phase}: training agent {label} for {this_phase_steps} steps")
        print(f"  (trainer.state.global_step={active_trainer.state.global_step} "
              f"→ max_steps={active_trainer.args.max_steps})")
        print(f"{'='*60}")

        active_trainer.train(resume_from_checkpoint=False)

        total_done += this_phase_steps
        phase      += 1

        # Save checkpoint after each phase
        if total_done % args.save_steps == 0 or total_done >= args.total_steps:
            _save_models(trainer_a, trainer_b, run_dir,
                         "a_latest", "b_latest")

        try:
            _save_loss_curve(loss_history, loss_plot, args.model_a, args.model_b)
        except Exception as e:
            print(f"  [plot] WARNING: {e}")

    # --- Final save ----------------------------------------------------------
    _save_models(trainer_a, trainer_b, run_dir, "a_final", "b_final")


def _save_models(trainer_a, trainer_b, run_dir, label_a, label_b):
    for trainer, label in [(trainer_a, label_a), (trainer_b, label_b)]:
        ckpt = os.path.join(run_dir, f"model_{label}")
        trainer.model.save_pretrained(ckpt)
        trainer.processing_class.save_pretrained(ckpt)
        print(f"  Saved model_{label} → {ckpt}")


# ---------------------------------------------------------------------------
# 13.  Evaluation helpers
# ---------------------------------------------------------------------------

def _run_eval(model_a, model_b, tok_a, tok_b, args, device, n=20):
    model_a.eval(); model_b.eval()
    wins = 0; rounds = 0
    nouns = get_concrete_nouns()
    print(f"\n=== Evaluation ({n} episodes) ===")
    for _ in range(n):
        sa, sb = random.sample(nouns, 2)
        ep = run_episode(model_a, model_b, tok_a, tok_b, sa, sb, device,
                         max_rounds=args.max_rounds, temperature=0.01,
                         bridge_threshold=args.bridge_threshold)
        wins += ep.won; rounds += ep.n_rounds
        _print_episode(ep)
    print(f"\nWin rate: {wins}/{n} = {wins/n:.1%}")
    print(f"Avg rounds: {rounds/n:.2f}")

def _log_episode_to_file(ep, step, log_path, ts):
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n[{ts}] step={step}  '{ep.start_a}' ↔ '{ep.start_b}'\n{'='*60}\n")
        f.write(f"  {'Round':<8} {'Agent A':<24} {'Agent B':<24}\n  {'-'*56}\n")
        for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
            ra = f"[r{len(ta.attempts)-1}]" if len(ta.attempts) > 1 else ""
            rb = f"[r{len(tb.attempts)-1}]" if len(tb.attempts) > 1 else ""
            ca = f"{ta.word}{ra}({'✓' if ta.is_valid_bridge else '✗'})"
            cb = f"{tb.word}{rb}({'✓' if tb.is_valid_bridge else '✗'})"
            f.write(f"  {i+1:<8} {ca:<24} {cb:<24}\n")
        outcome = "WON" if ep.won else ("EXHAUSTED" if ep.exhausted else "TIMEOUT")
        f.write(f"  → {outcome} in {ep.n_rounds} round(s)\n")

def _print_episode(ep):
    print(f"\n  Episode: '{ep.start_a}' ↔ '{ep.start_b}'")
    for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
        ra = f"[r{len(ta.attempts)-1}]" if len(ta.attempts) > 1 else ""
        rb = f"[r{len(tb.attempts)-1}]" if len(tb.attempts) > 1 else ""
        print(f"    Round {i+1}: A={ta.word}{ra}({'✓' if ta.is_valid_bridge else '✗'}) "
              f"B={tb.word}{rb}({'✓' if tb.is_valid_bridge else '✗'})")
    print(f"  → {'WON' if ep.won else 'EXHAUSTED' if ep.exhausted else 'TIMEOUT'} "
          f"in {ep.n_rounds} round(s)")

def _save_loss_curve(history, path, model_a="", model_b=""):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history: return

    nan = float("nan")
    def short(n): return n.split("/")[-1] if n else n
    label_a = f"{short(model_a)} (A)" if model_a else "loss_a"
    label_b = f"{short(model_b)} (B)" if model_b else "loss_b"

    # Separate series: each history entry has either loss_a or loss_b (not both)
    steps_a = [r["step"]   for r in history if r.get("loss_a") is not None]
    loss_a  = [r["loss_a"] for r in history if r.get("loss_a") is not None]
    steps_b = [r["step"]   for r in history if r.get("loss_b") is not None]
    loss_b  = [r["loss_b"] for r in history if r.get("loss_b") is not None]

    all_steps  = [r["step"]              for r in history]
    win_rates  = [r.get("win_rate",  nan) for r in history]
    avg_rounds = [r.get("avg_rounds", nan) for r in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps_a, loss_a, label=label_a, color="steelblue",  linewidth=1.2)
    ax1.plot(steps_b, loss_b, label=label_b, color="darkorange", linewidth=1.2)
    ax1.set_ylabel("GRPO loss")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(all_steps, win_rates, label="win_rate", color="green", linewidth=1.2)
    ax2.set_ylabel("win rate")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(all_steps, avg_rounds, label="avg rounds", color="purple",
              linewidth=1.2, linestyle="--", alpha=0.85)
    ax2r.set_ylabel("avg rounds to converge")
    ax2r.legend(loc="upper right")

    ax2.set_xlabel("step")
    fig.suptitle("GRPO Converge — TRL alternating agents — training curves")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
    print(f"  [plot] Loss curve saved → {path}")


# ---------------------------------------------------------------------------
# 14.  Args + entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a",             default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--model_b",             default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir",          default="./grpo_converge_output")
    p.add_argument("--max_rounds",          type=int,   default=10)
    p.add_argument("--bridge_threshold",    type=float, default=0.2)
    p.add_argument("--temperature",         type=float, default=0.9)
    p.add_argument("--num_groups",          type=int,   default=4)
    p.add_argument("--plays_per_group",     type=int,   default=8)
    p.add_argument("--eval_only",           action="store_true")
    p.add_argument("--eval_episodes",       type=int,   default=5)
    p.add_argument("--total_steps",         type=int,   default=500)
    p.add_argument("--learning_rate",       type=float, default=5e-6)
    p.add_argument("--kl_coef",             type=float, default=0.04)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Unused — TRL uses num_groups as grad_accum internally.")
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--swap_every",          type=int,   default=16,
                   help="Optimizer steps between alternating which agent trains.")
    p.add_argument("--use_peft",            action="store_true")
    p.add_argument("--lora_r",              type=int,   default=16)
    p.add_argument("--lora_alpha",          type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not WORDNET_AVAILABLE:
        print("ERROR: pip install nltk && python -c \"import nltk; "
              "nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        exit(1)
    train(args)

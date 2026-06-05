"""
Multi-agent GRPO training for the improv game "Converge" — leaf-noun variant.

Identical to grpo_converge_optimized.py except the word pool is restricted to
WordNet *leaf* nouns: the most specific, unambiguous concrete nouns (zero
hyponyms, primary sense also a leaf in a concrete lexname category). This
prevents models from trivially converging via broad category words like
"animal" or "plant" that sit at interior nodes of the hierarchy.

Leaf selection criteria:
  - Synset has zero hyponyms (true leaf in the hierarchy)
  - Primary WordNet sense for the word is also a concrete leaf
  - Brown corpus frequency ≥ 3 (filters out obscure / hapax words)
  - Not a proper noun (Brown mid-sentence capitalisation heuristic)
  - Not in the explicit blocklist (jargon, inappropriate, ambiguous)
  - Length 4–12 chars, alpha only

Usage:
    # Smoke test (CPU, no training)
    python grpo_converge_leaves.py --eval_only --eval_episodes 5

    # Full training (single GPU, LoRA)
    python grpo_converge_leaves.py --use_peft

    # Two different base checkpoints
    python grpo_converge_leaves.py --model_a meta-llama/Llama-3.2-3B-Instruct \\
                                    --model_b Qwen/Qwen2.5-3B-Instruct \\
                                    --use_peft
"""

import re
import random
import argparse
import string
import os
from collections import defaultdict
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

# ---------------------------------------------------------------------------
# 1.  Bridge validity
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
    sim_a = _best_sim(word, anchor_a)
    sim_b = _best_sim(word, anchor_b)
    return sim_a >= threshold and sim_b >= threshold

def is_in_wordnet(word: str) -> bool:
    return bool(wn.synsets(word, pos=wn.NOUN))

# ---------------------------------------------------------------------------
# 2.  Leaf-noun pool
# ---------------------------------------------------------------------------

# noun.object dropped — mostly geographic features / proper nouns
CONCRETE_LEXNAMES = {
    "noun.animal", "noun.plant", "noun.food", "noun.artifact",
    "noun.body", "noun.substance",
}

BLOCKLIST = {
    # inappropriate
    "ass", "dick", "cock", "bum", "butt", "buns", "tit", "anus", "rectum",
    "penis", "vagina", "scrotum", "vulva", "nipple", "groin", "urethra",
    "crotch", "labia", "foreskin", "glans", "stool", "urine", "feces",
    "crap", "crud", "muck", "slime", "gunk", "goo", "behind", "posterior",
    # people/mankind
    "humans", "mankind", "humanity", "person", "people",
    # chemistry / scientific jargon
    "anionic", "conjugate", "metabolite", "dilution", "substrate",
    "effluent", "volatile", "alkane", "alkene", "alkyne", "aldehyde",
    "ketone", "amine", "amide", "halide", "ether", "oxide", "sulfide",
    "nitrate", "sulfate", "phosphate", "carbonate", "hydroxide",
    "polymer", "monomer", "colloid", "emulsion", "photocathode",
    "phosphor", "urethane", "beebread", "glycerine", "saline",
    "gallium", "germanium", "argon", "helium", "neon", "potassium",
    "sodium", "thinner", "render", "rinse", "concentrate", "intermediate",
    "fallout", "fumes", "refuse", "compost", "sewage",
    "acetone", "acrylic", "aggregate", "alligator", "asphalt",
    "coolant", "corrosive", "digestive", "dross", "ethyl", "excelsior",
    "floc", "fraction", "ozone", "prednisone", "resorcinol", "retinal",
    "strychnine", "thiouracil", "tyrosine", "mecholyl", "laudanum",
    # primarily adjectives / relational
    "active", "simple", "technical", "metallic", "reversible", "portable",
    "electric", "duplicate", "compact", "limited", "graphic", "dietary",
    "total", "empty", "local", "lower", "upper", "short", "working",
    "workings", "ways", "fore", "grab", "etcetera", "tore",
    # process / gerund nouns
    "grinding", "scraping", "stopping", "leading",
    # obscure plant names
    "feathertop", "pseudophloem", "sang", "clocks", "glad", "rape",
    "benny", "cheat", "chess", "flamboyant", "melissa", "psyllium",
    "weld", "xylem", "yarrow", "larkspur",
    # obscure body / medical
    "epiphysis", "neocortex", "hypothalamus", "pituitary", "thyroxine",
    "phalanx", "quick", "small", "sweat", "permanent",
    "antiserum", "bronchiole", "bronchus", "collagen", "crus",
    "hilum", "intima", "jejunum", "parenchyma",
    # obscure animal / scientific names
    "bombus", "chen", "regulus", "midge", "ephemeral", "saratoga",
    "skipjack", "dipper", "horseback", "newfoundland",
    # misc junk
    "twins", "thus", "till", "otter", "keelson", "tappet", "quirt",
    "servo", "sounder", "sonar", "pegboard", "hibachi",
    "swivel", "transom", "shook", "weeds", "protozoa",
    "roughcast", "planking", "overlapping", "overload", "bombproof",
    "babel", "backstitch", "armhole", "curbside",
    # acronyms / proper
    "ada", "des", "mao", "pbs", "abs", "ace",
    "angel", "saint", "bishop", "affirmed", "citation", "imperial",
    "vatican",
}

_LEAF_NOUN_CACHE: list[str] = []

def get_leaf_nouns() -> list[str]:
    global _LEAF_NOUN_CACHE
    if _LEAF_NOUN_CACHE:
        return _LEAF_NOUN_CACHE

    from nltk.corpus import brown

    # Build Brown corpus frequency and capitalisation tables
    cap_freq:   dict[str, int] = defaultdict(int)
    lower_freq: dict[str, int] = defaultdict(int)
    for sent in brown.sents():
        for i, tok in enumerate(sent):
            w = tok.lower()
            if not re.fullmatch(r"[a-z]{4,12}", w):
                continue
            if i > 0 and tok[0].isupper():
                cap_freq[w] += 1
            else:
                lower_freq[w] += 1

    total_freq: dict[str, int] = defaultdict(int)
    for w, c in cap_freq.items():   total_freq[w] += c
    for w, c in lower_freq.items(): total_freq[w] += c

    def is_proper(w: str) -> bool:
        c, l = cap_freq[w], lower_freq[w]
        total = c + l
        if total < 3:
            return False
        return c / total > 0.5

    def has_instance_hypernym(synset) -> bool:
        return bool(synset.instance_hypernyms())

    leaf_by_cat: dict[str, set] = defaultdict(set)

    for synset in wn.all_synsets(pos=wn.NOUN):
        lex = synset.lexname()
        if lex not in CONCRETE_LEXNAMES:
            continue
        if len(synset.hyponyms()) > 0:
            continue
        for lemma in synset.lemmas():
            w = lemma.name().lower().replace("_", "")
            if not re.fullmatch(r"[a-z]{4,12}", w):
                continue
            if total_freq[w] < 3:
                continue
            if is_proper(w):
                continue
            if w in BLOCKLIST:
                continue
            if has_instance_hypernym(synset):
                continue

            # Primary sense must also be a concrete leaf
            primary = wn.synsets(w, pos=wn.NOUN)
            if not primary:
                continue
            p = primary[0]
            if p.lexname() not in CONCRETE_LEXNAMES:
                continue
            if len(p.hyponyms()) > 0:
                continue

            leaf_by_cat[lex].add(w)

    words = sorted({w for cat in leaf_by_cat.values() for w in cat})
    _LEAF_NOUN_CACHE = words
    print(f"[WordNet] Loaded {len(words)} leaf nouns across {len(leaf_by_cat)} categories.")
    return words

# ---------------------------------------------------------------------------
# 3.  Prompt
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
            eos_token_id=[
                tokenizer.eos_token_id,
                *tokenizer.encode("\n", add_special_tokens=False),
            ],
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
                   adapter_name: "str | None" = None) -> tuple[str, str, float, list[tuple[str,str]], bool]:
    all_attempts = []
    penalty = 0.0

    for attempt_idx in range(MAX_RETRIES):
        raw, prompt_text = _generate_once(
            model, tokenizer, messages, device, temperature,
            adapter_name=adapter_name,
        )
        all_attempts.append((prompt_text, raw))

        word = sanitize_output(raw)

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

    last_word = sanitize_output(all_attempts[-1][1]) or "unknown"
    return last_word, all_attempts[-1][0], penalty, all_attempts, True


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
                start_a: str, start_b: str,
                device: str, max_rounds: int = 10,
                temperature: float = 0.9,
                bridge_threshold: float = 0.10,
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
            adapter_name=adapter_name_a,
        )

        msgs_b = build_messages(prev_a, prev_b, history, "B", start_a, start_b)
        word_b, _, penalty_b, attempts_b, exhausted_b = get_valid_word(
            model_b, tokenizer_b, msgs_b, device, temperature,
            prev_a, prev_b, used_words, used_stems, _stem, bridge_threshold,
            adapter_name=adapter_name_b,
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


# ---------------------------------------------------------------------------
# 8.  Plays → Rollout conversion
# ---------------------------------------------------------------------------

def plays_to_rollout(plays: list[Episode], agent: str, group_id: int) -> Rollout:
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
        turns_to_include = turns[:-1] if ep.exhausted else turns
        for turn in turns_to_include:
            prompt_text, raw_output = turn.attempts[-1]
            prompts.append(prompt_text)
            completions.append(raw_output)
            rewards.append(adv)

    return Rollout(prompts=prompts, completions=completions,
                   rewards=rewards, group_id=group_id)


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------

def _make_run_dir(args) -> str:
    def short(name):
        return name.split("/")[-1]

    model_tag = (
        short(args.model_a) if args.model_a == args.model_b
        else f"{short(args.model_a)}_vs_{short(args.model_b)}"
    )
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{model_tag}_lr{args.learning_rate}_kl{args.kl_coef}"
        f"_B{args.num_groups}G{args.plays_per_group}"
        f"_steps{args.total_steps}_{ts}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    return run_dir, run_name


def train(args):
    if not WORDNET_AVAILABLE:
        raise ImportError(
            "pip install nltk && python -c \"import nltk; "
            "nltk.download('wordnet'); nltk.download('omw-1.4'); "
            "nltk.download('brown')\""
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    leaf_nouns = get_leaf_nouns()

    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_a.eos_token is None:
        tokenizer_a.eos_token = tokenizer_a.pad_token
    tokenizer_a.padding_side = "left"

    if args.model_b == args.model_a:
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

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir, run_name = _make_run_dir(args)
    game_log_path  = os.path.join(run_dir, "game_log.txt")
    loss_plot_path = os.path.join(run_dir, "loss_curve.png")
    loss_history: list[dict] = []

    with open(game_log_path, "a") as f:
        f.write(f"\n{'#'*70}\n")
        f.write(f"# NEW RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# model_a={args.model_a}\n")
        f.write(f"# model_b={args.model_b}\n")
        f.write(f"# lr={args.learning_rate}  kl_coef={args.kl_coef}  "
                f"num_groups={args.num_groups}  plays_per_group={args.plays_per_group}\n")
        f.write(f"# max_rounds={args.max_rounds}  bridge_threshold={args.bridge_threshold}  "
                f"temperature={args.temperature}\n")
        f.write(f"# total_steps={args.total_steps}  grad_accum={args.gradient_accumulation_steps}\n")
        f.write(f"# use_peft={args.use_peft}  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}\n")
        f.write(f"# word_pool=leaf_nouns ({len(leaf_nouns)} words)\n")
        f.write(f"# run_dir={run_dir}\n")
        f.write(f"{'#'*70}\n")

    global_step = 0

    for batch_idx in range(args.total_steps):

        starts = []
        while len(starts) < args.num_groups:
            a, b = random.sample(leaf_nouns, 2)
            if _best_path_sim(a, b) < 0.5:
                starts.append((a, b))

        model_a.eval(); model_b.eval()
        all_plays = []
        for (sa, sb) in starts:
            group_plays = []
            for _ in range(args.plays_per_group):
                ep = run_episode(
                    model_a, model_b, tokenizer_a, tokenizer_b,
                    sa, sb, device,
                    max_rounds=args.max_rounds,
                    temperature=args.temperature,
                    bridge_threshold=args.bridge_threshold,
                    adapter_name_a=adapter_name_a,
                    adapter_name_b=adapter_name_b,
                )
                group_plays.append(ep)
            all_plays.append(group_plays)

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

        if args.eval_only:
            for ep in flat[:2]:
                _print_episode(ep)
            global_step += 1
            continue

        rollouts_a = [plays_to_rollout(group, "a", gid)
                      for gid, group in enumerate(all_plays)]
        rollouts_b = [plays_to_rollout(group, "b", gid)
                      for gid, group in enumerate(all_plays)]

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

    ax1.plot(steps, loss_a, label="loss_a", color="steelblue",  linewidth=1.2)
    ax1.plot(steps, loss_b, label="loss_b", color="darkorange", linewidth=1.2)
    ax1.set_ylabel("GRPO loss")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

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

    fig.suptitle("GRPO Converge Leaves — training curves")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Loss curve saved → {plot_path}")

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
    outcome = "WON" if ep.won else ("EXHAUSTED" if ep.exhausted else "TIMEOUT")
    print(f"  → {outcome} in {ep.n_rounds} round(s)")

def evaluate(model_a, model_b, tokenizer_a, tokenizer_b,
             args, n_episodes: int = 20):
    device = next(model_a.parameters()).device
    model_a.eval(); model_b.eval()

    wins = 0; total_rounds = 0
    leaf_nouns = get_leaf_nouns()
    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    for _ in range(n_episodes):
        sa, sb = random.sample(leaf_nouns, 2)
        ep = run_episode(
            model_a, model_b, tokenizer_a, tokenizer_b,
            sa, sb, str(device),
            max_rounds=args.max_rounds,
            temperature=0.01,
            bridge_threshold=args.bridge_threshold,
        )
        wins += ep.won
        total_rounds += ep.n_rounds
        _print_episode(ep)

    print(f"\nWin rate : {wins}/{n_episodes} = {wins/n_episodes:.1%}")
    print(f"Avg rounds: {total_rounds/n_episodes:.2f}")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a",              default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model_b",              default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir",           default="./grpo_converge_leaves_output")
    p.add_argument("--max_rounds",           type=int,   default=10)
    p.add_argument("--bridge_threshold",     type=float, default=0.2)
    p.add_argument("--temperature",          type=float, default=0.9)
    p.add_argument("--num_groups",           type=int,   default=4)
    p.add_argument("--plays_per_group",      type=int,   default=8)
    p.add_argument("--eval_only",            action="store_true")
    p.add_argument("--eval_episodes",        type=int,   default=5)
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
              "nltk.download('omw-1.4'); nltk.download('brown')\"")
        exit(1)

    train(args)

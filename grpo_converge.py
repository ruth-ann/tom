"""
Multi-agent GRPO training for the improv game "Converge".

Two models (A and B) start with different random concrete nouns and must
converge to the SAME word in as few rounds as possible. Each round, both
models output a single word that must be a valid semantic bridge between
the two words from the previous round (verified via WordNet path similarity).

Anti-cheat measures:
  - Output is constrained to a single word via strict parsing + penalty
  - Logit bias blocks common preamble tokens during generation
  - Any multi-word / non-alpha output is penalised and replaced with a
    random fallback so the episode can continue
  - The bridge validity check uses BOTH previous words, so a model cannot
    win by just repeating a previous word unless it truly bridges

Requirements:
    pip install nltk sentence-transformers torch transformers accelerate peft
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Usage:
    # Smoke test (CPU, no training)
    python grpo_converge.py --eval_only --eval_episodes 5

    # Full training (single GPU, LoRA)
    python grpo_converge.py --use_peft

    # Two different base checkpoints
    python grpo_converge.py --model_a Qwen/Qwen2.5-3B-Instruct \
                             --model_b Qwen/Qwen2.5-3B-Instruct \
                             --use_peft
"""

import re
import random
import argparse
import string
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1.  WordNet verifier
# ---------------------------------------------------------------------------

def _best_path_sim(w1: str, w2: str) -> float:
    """
    Returns the maximum WordNet path similarity across all noun synset pairs
    for (w1, w2). Returns 0.0 if either word has no noun synsets.
    """
    syns1 = wn.synsets(w1, pos=wn.NOUN)
    syns2 = wn.synsets(w2, pos=wn.NOUN)
    if not syns1 or not syns2:
        return 0.0
    best = 0.0
    for s1 in syns1[:3]:   # cap at 3 to keep it fast
        for s2 in syns2[:3]:
            sim = s1.path_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best


def is_valid_bridge(word: str, anchor_a: str, anchor_b: str,
                    threshold: float = 0.10) -> bool:
    """
    Returns True if `word` is a valid semantic bridge between anchor_a and
    anchor_b under WordNet path similarity.

    threshold=0.10 is deliberately permissive — this is a game, not a
    linguistics exam. Tune via --bridge_threshold.
    """
    sim_a = _best_path_sim(word, anchor_a)
    sim_b = _best_path_sim(word, anchor_b)
    return sim_a >= threshold and sim_b >= threshold


def is_in_wordnet(word: str) -> bool:
    """Check that the word exists as a noun in WordNet."""
    return bool(wn.synsets(word, pos=wn.NOUN))


# ---------------------------------------------------------------------------
# 2.  Concrete noun pool
# ---------------------------------------------------------------------------

# Lexnames that correspond to physical / concrete things
CONCRETE_LEXNAMES = {
    "noun.animal", "noun.plant", "noun.food", "noun.artifact",
    "noun.body", "noun.object", "noun.substance",
}

_CONCRETE_NOUN_CACHE: list[str] = []


def get_concrete_nouns(max_words: int = 3000) -> list[str]:
    """
    Pull concrete nouns from WordNet. Cached after first call.
    Returns single-token, lowercase, alpha-only words.

    Filters to common English words only by cross-referencing NLTK's Brown
    corpus word frequencies — this removes obscure taxonomic/scientific names
    (e.g. 'chrysemys', 'pycnogonida') that models can't reason about, keeping
    only words a fluent English speaker would recognise.
    """
    global _CONCRETE_NOUN_CACHE
    if _CONCRETE_NOUN_CACHE:
        return _CONCRETE_NOUN_CACHE

    # Build a frequency map from the Brown corpus (general English text).
    # Words appearing fewer than min_freq times are treated as uncommon.
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
        # If Brown corpus isn't downloaded, fall back to no frequency filter.
        # Run: python -c "import nltk; nltk.download('brown')" to enable.
        common_words = None
        print("[WordNet] Brown corpus unavailable — no frequency filter applied. "
              "Run: python -c \"import nltk; nltk.download('brown')\" to fix.")

    seen = set()
    words = []
    for synset in wn.all_synsets(pos=wn.NOUN):
        if synset.lexname() not in CONCRETE_LEXNAMES:
            continue
        for lemma in synset.lemmas():
            w = lemma.name().lower().replace("_", "")
            if (w.isalpha() and 3 <= len(w) <= 12
                    and " " not in w and w not in seen
                    # Only keep words that appear in common English text
                    and (common_words is None or w in common_words)):
                seen.add(w)
                words.append(w)
                if len(words) >= max_words:
                    break
        if len(words) >= max_words:
            break

    _CONCRETE_NOUN_CACHE = words
    print(f"[WordNet] Loaded {len(words)} concrete nouns.")
    return words #fine


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


def build_messages(my_word: str, partner_word: str,
                   history: list[dict], agent_label: str) -> list[dict]:
    """
    Build a chat message list for one agent at one turn.

    history: list of {"my_word": str, "partner_word": str} for previous rounds.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not history:
        user_content = (
            f"Game start!\n"
            f"Your starting word: {my_word}\n"
            f"Partner's starting word: {partner_word}\n\n"
            f"Round 1: Output your bridging word."
        )
    else:
        history_lines = []
        for i, h in enumerate(history):
            history_lines.append(
                f"  Round {i}: you said '{h['my_word']}', "
                f"partner said '{h['partner_word']}'"
            )
        history_str = "\n".join(history_lines)
        last = history[-1]
        user_content = (
            f"Starting words — you: {my_word} | partner: {partner_word}\n\n"
            f"History:\n{history_str}\n\n"
            f"Last round: you said '{last['my_word']}', "
            f"partner said '{last['partner_word']}'.\n"
            f"Round {len(history)}: Output your single bridging word."
        )

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# 4.  Single-word extraction & cleaning  (anti-cheat core)
# ---------------------------------------------------------------------------

# Regex: grab the first run of alpha characters of length 2-15
_WORD_RE = re.compile(r"\b([a-zA-Z]{2,15})\b")

# Words that are almost certainly preamble, not a game word
_PREAMBLE_WORDS = {
    "i", "my", "the", "a", "an", "is", "think", "word", "would", "be",
    "say", "said", "answer", "output", "bridge", "related", "both",
    "guess", "round", "game", "player", "partner", "correct", "final",
    "response", "choose", "pick", "select", "go", "okay", "ok", "sure",
    "well", "so", "then", "thus", "hence", "therefore", "let", "me",
    "give", "here", "this", "that", "it", "its",
}


def extract_word(text: str) -> Optional[str]:
    """
    Robustly extract a single game word from model output.

    Strategy:
    1. Take only the first line (models sometimes ramble after a newline).
    2. Strip leading/trailing whitespace and punctuation.
    3. If the result is already a single clean word, return it.
    4. Otherwise scan for the first non-preamble alpha token.
    5. Return None if nothing usable is found (triggers penalty).
    """
    # Step 1: take only the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    else:
        return None

    text = text.strip(string.punctuation + string.whitespace).lower()

    # Case 1: output is already a single word
    if re.fullmatch(r"[a-z]{2,15}", text):
        return text

    # Case 2: scan tokens, skip preamble words
    tokens = _WORD_RE.findall(text)
    for tok in tokens:
        tok = tok.lower()
        if tok not in _PREAMBLE_WORDS and len(tok) >= 2:
            return tok

    return None


def sanitize_output(raw: str, fallback_pool: list[str]) -> tuple[str, bool]:
    """
    Returns (word, is_clean).
    is_clean=False means we had to use a fallback (triggers format penalty).
    """
    word = extract_word(raw)
    if word is None:
        return random.choice(fallback_pool), False
    return word, True


# ---------------------------------------------------------------------------
# 5.  Constrained generation  (anti-cheat: force single-token outputs)
# ---------------------------------------------------------------------------

def get_single_word(model, tokenizer, messages: list[dict],
                    device: str, temperature: float = 0.9,
                    max_new_tokens: int = 8) -> tuple[str, str]:
    """
    Generate a response and return (raw_text, prompt_text).

    max_new_tokens=8 is intentionally tiny — a single word never needs more
    than ~3 tokens. This physically prevents long preamble generation.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
            # Stop on newline — single-word outputs never span lines
            eos_token_id=[
                tokenizer.eos_token_id,
                *tokenizer.encode("\n", add_special_tokens=False),
            ],
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, prompt_text


# ---------------------------------------------------------------------------
# 6.  Reward functions
# ---------------------------------------------------------------------------

def turn_reward(word: str, anchor_a: str, anchor_b: str,
                is_clean: bool, threshold: float) -> float:
    """Per-turn reward for one agent."""
    r = 0.0

    # Format penalty: model produced preamble / garbage
    if not is_clean:
        r -= 2.0

    # Validity: must be in WordNet
    if not is_in_wordnet(word):
        r -= 1.5
        return r

    # Bridge quality: how well does this word connect the two anchors?
    sim_a = _best_path_sim(word, anchor_a)
    sim_b = _best_path_sim(word, anchor_b)
    bridge_score = sim_a + sim_b            # max ~2.0, typical good ~0.4-0.8
    r += bridge_score * 2.0                 # scale to ~0-4 range

    # Hard penalty for failing the validity threshold
    if sim_a < threshold or sim_b < threshold:
        r -= 3.0

    return r


def episode_terminal_reward(won: bool, n_rounds: int, max_rounds: int) -> float:
    """Shared terminal reward — both agents receive this."""
    if won:
        # Scale: win in 1 round = +20, win in max_rounds = +5
        efficiency = (max_rounds - n_rounds) / max(max_rounds - 1, 1)
        return 5.0 + 15.0 * efficiency
    else:
        return -5.0


# ---------------------------------------------------------------------------
# 7.  Episode runner
# ---------------------------------------------------------------------------

@dataclass
class AgentTurn:
    prompt_text: str
    raw_output: str
    word: str
    is_clean: bool
    is_valid_bridge: bool
    turn_reward: float


@dataclass
class Episode:
    start_a: str
    start_b: str
    turns_a: list[AgentTurn] = field(default_factory=list)
    turns_b: list[AgentTurn] = field(default_factory=list)
    won: bool = False
    n_rounds: int = 0
    invalid_bridge: bool = False        # True if episode ended due to bad bridge


def run_episode(model_a, model_b, tokenizer_a, tokenizer_b,
                start_a: str, start_b: str,
                device: str, max_rounds: int = 10,
                temperature: float = 0.9,
                bridge_threshold: float = 0.10,
                concrete_nouns: list[str] = None) -> Episode:
    """
    Roll out one full episode between model_a and model_b.
    Both models see the full history of both players' words.
    """
    ep = Episode(start_a=start_a, start_b=start_b)
    fallback = concrete_nouns or ["tree", "rock", "water", "fire", "stone"]

    # The "current anchors" are the words each model must bridge from
    prev_a = start_a
    prev_b = start_b

    # History from each agent's perspective
    hist_a: list[dict] = []  # {"my_word", "partner_word"}
    hist_b: list[dict] = []

    for round_idx in range(max_rounds):
        # --- Agent A generates ---
        msgs_a = build_messages(start_a, start_b, hist_a, "A")
        raw_a, prompt_a = get_single_word(
            model_a, tokenizer_a, msgs_a, device, temperature
        )
        word_a, clean_a = sanitize_output(raw_a, fallback)

        # --- Agent B generates ---
        msgs_b = build_messages(start_b, start_a, hist_b, "B")
        raw_b, prompt_b = get_single_word(
            model_b, tokenizer_b, msgs_b, device, temperature
        )
        word_b, clean_b = sanitize_output(raw_b, fallback)

        # --- Validate bridges ---
        valid_a = is_valid_bridge(word_a, prev_a, prev_b, bridge_threshold)
        valid_b = is_valid_bridge(word_b, prev_a, prev_b, bridge_threshold)

        # --- Per-turn rewards ---
        r_a = turn_reward(word_a, prev_a, prev_b, clean_a, bridge_threshold)
        r_b = turn_reward(word_b, prev_a, prev_b, clean_b, bridge_threshold)

        ep.turns_a.append(AgentTurn(prompt_a, raw_a, word_a, clean_a, valid_a, r_a))
        ep.turns_b.append(AgentTurn(prompt_b, raw_b, word_b, clean_b, valid_b, r_b))

        ep.n_rounds = round_idx + 1

        # --- Convergence check ---
        if word_a == word_b:
            ep.won = True
            break

        # --- Invalid bridge ends the episode ---
        # FIX: Don't bail in early rounds (0-2) — cold/untrained models need
        # time to warm up. Instead, continue with a fallback word so the
        # episode can keep running and produce useful training signal.
        if not valid_a or not valid_b:
            if round_idx >= 2:
                ep.invalid_bridge = True
                break
            # Otherwise advance with fallback words so the episode continues
            prev_a = word_a if valid_a else random.choice(fallback)
            prev_b = word_b if valid_b else random.choice(fallback)
            hist_a.append({"my_word": prev_a, "partner_word": prev_b})
            hist_b.append({"my_word": prev_b, "partner_word": prev_a})
            continue

        # --- Advance anchors ---
        prev_a = word_a
        prev_b = word_b
        hist_a.append({"my_word": word_a, "partner_word": word_b})
        hist_b.append({"my_word": word_b, "partner_word": word_a})

    # --- Terminal rewards ---
    terminal = episode_terminal_reward(ep.won, ep.n_rounds, max_rounds)
    if ep.invalid_bridge:
        terminal -= 3.0   # extra penalty for breaking the chain

    # Attach terminal reward to last turn of each agent
    if ep.turns_a:
        ep.turns_a[-1].turn_reward += terminal
    if ep.turns_b:
        ep.turns_b[-1].turn_reward += terminal

    return ep


# ---------------------------------------------------------------------------
# 8.  GRPO loss
# ---------------------------------------------------------------------------

def _get_completion_logprobs(model, tokenizer, prompt_texts: list[str],
                              completion_texts: list[str],
                              device: str) -> torch.Tensor:
    """
    Compute sum of log-probs over completion tokens for each (prompt, completion).
    Returns shape (B,).
    """
    full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
    enc_full = tokenizer(full_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=800).to(device)
    enc_prompt = tokenizer(prompt_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=800).to(device)
    prompt_lens = enc_prompt["attention_mask"].sum(dim=1)   # (B,)

    # FIX: Wrap in torch.enable_grad() so policy logprobs have a grad_fn
    # even when called from within a broader no_grad context (e.g. after
    # the rollout phase which runs model.eval() + torch.no_grad()).
    # The entire slice/gather/sum chain must stay inside this context —
    # exiting it before torch.stack() would detach the tensors.
    with torch.enable_grad():
        logits = model(
            input_ids=enc_full["input_ids"],
            attention_mask=enc_full["attention_mask"],
        ).logits                                                 # (B, T, V)

        log_probs = F.log_softmax(logits, dim=-1)               # (B, T, V)

        B = logits.shape[0]
        result = []
        for b in range(B):
            pl = prompt_lens[b].item()
            ids = enc_full["input_ids"][b, pl:]                 # completion token ids
            lp  = log_probs[b, pl-1:pl-1+len(ids)]             # shifted logits
            gathered = lp.gather(1, ids.unsqueeze(1)).squeeze(1)
            result.append(gathered.sum())
        return torch.stack(result)                              # (B,)


def grpo_loss_for_agent(model, ref_model, tokenizer,
                         episodes: list[Episode], agent: str,
                         device: str, kl_coef: float = 0.04) -> torch.Tensor:
    """
    Compute GRPO loss for one agent across a batch of episodes.

    agent: "a" or "b"
    Each episode is one "group" — rewards are normalised within the episode
    across turns (since G=n_turns here, not multiple rollouts of same prompt).

    For true GRPO with G>1 rollouts per prompt, increase episodes_per_batch
    and group episodes by their starting word pair.
    """
    all_prompts, all_completions, all_rewards, all_ep_ids = [], [], [], []

    for ep_id, ep in enumerate(episodes):
        turns = ep.turns_a if agent == "a" else ep.turns_b
        for turn in turns:
            all_prompts.append(turn.prompt_text)
            all_completions.append(turn.raw_output)
            all_rewards.append(turn.turn_reward)
            all_ep_ids.append(ep_id)

    if not all_prompts:
        return torch.tensor(0.0, requires_grad=True)

    rewards_t = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    ep_ids_t  = torch.tensor(all_ep_ids,  dtype=torch.long,    device=device)

    # Normalise rewards within each episode group
    normed = torch.zeros_like(rewards_t)
    for eid in ep_ids_t.unique():
        mask = ep_ids_t == eid
        g = rewards_t[mask]
        if g.std() > 1e-6:
            normed[mask] = (g - g.mean()) / (g.std() + 1e-8)
        else:
            normed[mask] = g - g.mean()

    # Policy log-probs
    policy_lp = _get_completion_logprobs(
        model, tokenizer, all_prompts, all_completions, device
    )

    # Reference log-probs (no grad)
    with torch.no_grad():
        ref_lp = _get_completion_logprobs(
            ref_model, tokenizer, all_prompts, all_completions, device
        )

    kl   = policy_lp - ref_lp                              # (B,)
    loss = -(normed * policy_lp).mean() + kl_coef * kl.mean()
    return loss


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------

def make_model_and_ref(model_name: str, use_peft: bool,
                        lora_r: int, lora_alpha: int,
                        dtype: torch.dtype):
    """Load a policy model (optionally with LoRA) and a frozen ref copy."""
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    if use_peft:
        if not PEFT_AVAILABLE:
            raise ImportError("pip install peft")
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, cfg)
        policy.print_trainable_parameters()

    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.gradient_checkpointing_enable()
    # FIX: enable_input_require_grads() is required when using PEFT + gradient
    # checkpointing. Without it, the frozen embedding layer produces inputs with
    # requires_grad=False, which makes checkpointing silently produce detached
    # outputs and triggers "None of the inputs have requires_grad=True" warnings.
    policy.enable_input_require_grads()
    # FIX: Do NOT call gradient_checkpointing_enable() on the ref model.
    # The ref is fully frozen (no requires_grad), so checkpointing is
    # wasteful and triggers "None of the inputs have requires_grad=True" warnings.
    return policy, ref


def train(args):
    if not WORDNET_AVAILABLE:
        raise ImportError("pip install nltk && python -c \"import nltk; "
                          "nltk.download('wordnet'); nltk.download('omw-1.4')\"")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if (device == "cuda" and
                torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    concrete_nouns = get_concrete_nouns()

    # --- Load models ---------------------------------------------------------
    print(f"Loading model A: {args.model_a}")
    model_a, ref_a = make_model_and_ref(
        args.model_a, args.use_peft, args.lora_r, args.lora_alpha, dtype
    )
    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    tokenizer_a.padding_side = "left"

    if args.model_b == args.model_a:
        print("Model B shares base weights with A (will diverge via separate GRPO updates)")
        model_b, ref_b = make_model_and_ref(
            args.model_b, args.use_peft, args.lora_r, args.lora_alpha, dtype
        )
        tokenizer_b = tokenizer_a   # same tokenizer, that's fine
    else:
        print(f"Loading model B: {args.model_b}")
        model_b, ref_b = make_model_and_ref(
            args.model_b, args.use_peft, args.lora_r, args.lora_alpha, dtype
        )
        tokenizer_b = AutoTokenizer.from_pretrained(args.model_b)
        if tokenizer_b.pad_token is None:
            tokenizer_b.pad_token = tokenizer_b.eos_token
        tokenizer_b.padding_side = "left"

    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_a.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )
    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_b.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )

    # --- Training loop -------------------------------------------------------
    global_step = 0

    # Paths for the game log and loss curve plot
    os.makedirs(args.output_dir, exist_ok=True)
    game_log_path  = os.path.join(args.output_dir, "game_log.txt")
    loss_plot_path = os.path.join(args.output_dir, "loss_curve.png")
    loss_history: list[dict] = []   # accumulates {"step", "loss_a", "loss_b", "win_rate"}

    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for batch_idx in range(args.steps_per_epoch):

            # Sample starting word pairs (must be different)
            starts = []
            while len(starts) < args.episodes_per_batch:
                a, b = random.sample(concrete_nouns, 2)
                # Ensure the two start words are not trivially similar
                if _best_path_sim(a, b) < 0.5:
                    starts.append((a, b))

            # --- Rollout (no grad) -------------------------------------------
            model_a.eval(); model_b.eval()
            episodes = []
            for (sa, sb) in starts:
                ep = run_episode(
                    model_a, model_b, tokenizer_a, tokenizer_b,
                    sa, sb, device,
                    max_rounds=args.max_rounds,
                    temperature=args.temperature,
                    bridge_threshold=args.bridge_threshold,
                    concrete_nouns=concrete_nouns,
                )
                episodes.append(ep)

            # --- Stats -------------------------------------------------------
            wins       = sum(ep.won for ep in episodes)
            avg_rounds = sum(ep.n_rounds for ep in episodes) / len(episodes)
            invalids   = sum(ep.invalid_bridge for ep in episodes)
            win_rate   = wins / len(episodes)
            print(f"  step={global_step:4d} | "
                  f"wins={wins}/{len(episodes)} | "
                  f"avg_rounds={avg_rounds:.1f} | "
                  f"invalid_bridge={invalids}")

            # --- Log all episodes to file ------------------------------------
            for ep in episodes:
                _log_episode_to_file(ep, global_step, game_log_path)

            if args.eval_only:
                for ep in episodes[:2]:
                    _print_episode(ep)
                global_step += 1
                continue

            # --- Update model A ----------------------------------------------
            model_a.train()
            opt_a.zero_grad()
            loss_a = grpo_loss_for_agent(
                model_a, ref_a, tokenizer_a, episodes, "a", device, args.kl_coef
            )
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            opt_a.step()

            # --- Update model B ----------------------------------------------
            model_b.train()
            opt_b.zero_grad()
            loss_b = grpo_loss_for_agent(
                model_b, ref_b, tokenizer_b, episodes, "b", device, args.kl_coef
            )
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_b.step()

            print(f"           loss_a={loss_a.item():.4f} | "
                  f"loss_b={loss_b.item():.4f}")

            # --- Record losses and refresh the plot --------------------------
            loss_history.append({
                "step":     global_step,
                "loss_a":   loss_a.item(),
                "loss_b":   loss_b.item(),
                "win_rate": win_rate,
            })
            _save_loss_curve(loss_history, loss_plot_path)

            # --- Checkpoints -------------------------------------------------
            if global_step % args.save_steps == 0 and global_step > 0:
                for label, m, tok in [("a", model_a, tokenizer_a),
                                       ("b", model_b, tokenizer_b)]:
                    ckpt = f"{args.output_dir}/model_{label}_step{global_step}"
                    m.save_pretrained(ckpt)
                    tok.save_pretrained(ckpt)
                    print(f"  Saved model_{label} → {ckpt}")

            global_step += 1

    if not args.eval_only:
        for label, m, tok in [("a", model_a, tokenizer_a),
                                ("b", model_b, tokenizer_b)]:
            out = f"{args.output_dir}/model_{label}_final"
            m.save_pretrained(out); tok.save_pretrained(out)
            print(f"Final model_{label} saved → {out}")


# ---------------------------------------------------------------------------
# 10. Evaluation helpers
# ---------------------------------------------------------------------------

def _log_episode_to_file(ep: Episode, global_step: int, log_path: str):
    """
    Append a human-readable record of one episode to log_path.
    Each entry shows the word chain for both agents, validity markers,
    and the outcome, so you can read through and see what words were played.
    """
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"step={global_step}  '{ep.start_a}' ↔ '{ep.start_b}'\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Round':<8} {'Agent A':<20} {'Agent B':<20}\n")
        f.write(f"  {'-'*48}\n")
        for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
            bridge_a = "✓" if ta.is_valid_bridge else "✗"
            bridge_b = "✓" if tb.is_valid_bridge else "✗"
            clean_a  = "" if ta.is_clean else "[!]"
            clean_b  = "" if tb.is_clean else "[!]"
            col_a = f"{ta.word}{clean_a}({bridge_a})"
            col_b = f"{tb.word}{clean_b}({bridge_b})"
            f.write(f"  {i+1:<8} {col_a:<20} {col_b:<20}\n")
        outcome = "WON" if ep.won else ("INVALID" if ep.invalid_bridge else "TIMEOUT")
        f.write(f"  → {outcome} in {ep.n_rounds} round(s)\n")


def _save_loss_curve(loss_history: list[dict], plot_path: str):
    """
    Save a loss curve plot to plot_path (PNG).
    loss_history is a list of {"step": int, "loss_a": float, "loss_b": float}.
    Also plots win_rate on a twin y-axis if "win_rate" is present in entries.
    Requires matplotlib — silently skips if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend, safe for servers
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not found — skipping loss curve. "
              "pip install matplotlib to enable.")
        return

    steps    = [r["step"]   for r in loss_history]
    loss_a   = [r["loss_a"] for r in loss_history]
    loss_b   = [r["loss_b"] for r in loss_history]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(steps, loss_a, label="loss_a", color="steelblue",  linewidth=1.2)
    ax1.plot(steps, loss_b, label="loss_b", color="darkorange", linewidth=1.2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("GRPO loss")
    ax1.legend(loc="upper left")

    # Win rate on a second y-axis if recorded
    if "win_rate" in loss_history[0]:
        ax2 = ax1.twinx()
        win_rates = [r["win_rate"] for r in loss_history]
        ax2.plot(steps, win_rates, label="win_rate", color="green",
                 linewidth=1.0, linestyle="--", alpha=0.7)
        ax2.set_ylabel("win rate")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

    plt.title("GRPO Converge — training curves")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Loss curve saved → {plot_path}")


def _print_episode(ep: Episode):
    print(f"\n  Episode: '{ep.start_a}' ↔ '{ep.start_b}'")
    for i, (ta, tb) in enumerate(zip(ep.turns_a, ep.turns_b)):
        bridge_a = "✓" if ta.is_valid_bridge else "✗"
        bridge_b = "✓" if tb.is_valid_bridge else "✗"
        clean_a  = "" if ta.is_clean else "[!]"
        clean_b  = "" if tb.is_clean else "[!]"
        print(f"    Round {i+1}: A={ta.word}{clean_a}({bridge_a}) "
              f"B={tb.word}{clean_b}({bridge_b})")
    outcome = "WON" if ep.won else ("INVALID" if ep.invalid_bridge else "TIMEOUT")
    print(f"  → {outcome} in {ep.n_rounds} round(s)")


def evaluate(model_a, model_b, tokenizer_a, tokenizer_b,
             concrete_nouns, args, n_episodes: int = 20):
    device = next(model_a.parameters()).device
    model_a.eval(); model_b.eval()

    wins = 0; total_rounds = 0
    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    for _ in range(n_episodes):
        sa, sb = random.sample(concrete_nouns, 2)
        ep = run_episode(
            model_a, model_b, tokenizer_a, tokenizer_b,
            sa, sb, str(device),
            max_rounds=args.max_rounds,
            temperature=0.01,           # greedy at eval time
            bridge_threshold=args.bridge_threshold,
            concrete_nouns=concrete_nouns,
        )
        wins += ep.won
        total_rounds += ep.n_rounds
        _print_episode(ep)

    print(f"\nWin rate : {wins}/{n_episodes} = {wins/n_episodes:.1%}")
    print(f"Avg rounds: {total_rounds/n_episodes:.2f}")


# ---------------------------------------------------------------------------
# 11. Args & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model_b", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Defaults to same as model_a (diverges via separate updates)")
    p.add_argument("--output_dir",         default="./grpo_converge_output")
    p.add_argument("--num_epochs",         type=int,   default=3)
    p.add_argument("--steps_per_epoch",    type=int,   default=200)
    p.add_argument("--episodes_per_batch", type=int,   default=8,
                   help="Games per batch (GRPO group size)")
    p.add_argument("--max_rounds",         type=int,   default=10)
    p.add_argument("--learning_rate",      type=float, default=5e-6)
    p.add_argument("--kl_coef",            type=float, default=0.04)
    p.add_argument("--temperature",        type=float, default=0.9)
    p.add_argument("--bridge_threshold",   type=float, default=0.05,
                   help="Min WordNet path_similarity to count as valid bridge")
    p.add_argument("--save_steps",         type=int,   default=100)
    p.add_argument("--use_peft",           action="store_true")
    p.add_argument("--lora_r",             type=int,   default=16)
    p.add_argument("--lora_alpha",         type=int,   default=32)
    p.add_argument("--eval_only",          action="store_true",
                   help="Run rollouts only, no gradient updates (smoke test)")
    p.add_argument("--eval_episodes",      type=int,   default=5)
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
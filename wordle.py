"""
Multi-turn GRPO training for Wordle on a language model <= 8B.

The key idea: we roll out FULL 6-turn Wordle episodes, collect the entire
trajectory (prompt+completion at every turn), then assign rewards to each
turn and train with GRPO across the whole episode.

Requirements:
    pip install trl>=0.12.0 transformers>=4.46.0 accelerate peft datasets torch

Usage:
    # Single GPU with LoRA (recommended, needs ~16GB VRAM for 3B model)
    python grpo_wordle.py --use_peft

    # Larger model
    python grpo_wordle.py --model_name meta-llama/Llama-3.2-8B-Instruct --use_peft

    # Multi-GPU
    accelerate launch --num_processes 4 grpo_wordle.py --use_peft

    # Quick smoke-test (tiny dataset, 1 epoch)
    python grpo_wordle.py --use_peft --dataset_size 64 --num_train_epochs 1 --eval_only
"""

import re
import json
import random
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Word list
# ---------------------------------------------------------------------------
# A representative set of common 5-letter words.
# In production, replace with the full Wordle answer list (~2300 words).

WORDLE_WORDS = [
    "crane", "slate", "trace", "crate", "audio", "raise", "arise",
    "stare", "snare", "share", "shore", "store", "score", "scope",
    "space", "spare", "spore", "spoke", "smoke", "smote", "smile",
    "stale", "scale", "shale", "shake", "shame", "shape", "shade",
    "spade", "grade", "grape", "graze", "grace", "grave", "brave",
    "brace", "place", "plane", "plate", "plage", "blaze", "blade",
    "blame", "flame", "flare", "flake", "flute", "flume", "plume",
    "prune", "prone", "probe", "price", "pride", "prime", "grime",
    "gripe", "gripe", "tripe", "tribe", "bribe", "bride", "drive",
    "drove", "prove", "grove", "groan", "grown", "brown", "crown",
    "crowd", "croud", "cloud", "clown", "blown", "blood", "flood",
    "floor", "flour", "flout", "shout", "stout", "scout", "snout",
    "trout", "grout", "about", "above", "stove", "glove", "clove",
    "clone", "stone", "phone", "shown", "shone", "shoes", "chose",
    "close", "those", "prose", "arose", "house", "mouse", "grouse",
    "pause", "cause", "gauze", "gauge", "glare", "glaze", "glass",
    "class", "brass", "crass", "gross", "cross", "cress", "dress",
    "press", "chess", "bless", "flesh", "fresh", "trash", "crash",
    "brash", "clash", "flash", "flesh", "leash", "teach", "reach",
    "beach", "peach", "poach", "coach", "roach", "broach", "clomp",
    "stomp", "stump", "clump", "slump", "plump", "trump", "grump",
    "crimp", "skimp", "shrimp", "scrub", "shrub", "grub", "snub",
    "club", "slub", "stub", "snout", "spout", "stout", "clout",
    "trout", "route", "outer", "utter", "liter", "later", "water",
    "alter", "alert", "avert", "overt", "inert", "exert", "evert",
    "event", "evoke", "emote", "elate", "elite", "unite", "untie",
    "until", "optic", "orbit", "order", "other", "otter", "offer",
    "often", "occur", "ocean", "onion", "olive", "opera", "obese",
    "oxide", "ozone", "piano", "pilot", "pixel", "pixie", "pizza",
    "plaid", "plain", "plant", "plaza", "plead", "pleat", "plied",
    "plier", "pluck", "plunk", "polar", "polka", "polyp", "poppy",
    "porch", "power", "prank", "prawn", "privy", "proxy", "psalm",
    "pubic", "pudgy", "pulse", "puppy", "perch", "purse", "pushy",
    "pygmy", "quack", "quaff", "quail", "quake", "qualm", "quart",
    "quash", "quasi", "queen", "query", "queue", "quick", "quiet",
    "quill", "quirk", "quota", "quote", "rabbi", "radar", "radii",
    "rainy", "rally", "ramen", "ranch", "rapid", "raspy", "ravel",
    "raven", "rayon", "repay", "repel", "repot", "rerun", "resin",
    "retch", "rider", "ridge", "rigid", "rigor", "risky", "rivet",
    "robin", "rocky", "rodeo", "rogue", "roman", "roomy", "roost",
    "rowdy", "royal", "ruddy", "rugby", "ruler", "rumor", "rupee",
    "rusty", "sadly", "saint", "salad", "saline", "salon", "salsa",
    "sappy", "sassy", "sauce", "saucy", "sauna", "savor", "savoy",
    "savvy", "scald", "scalp", "scaly", "scamp", "scant", "scary",
    "scene", "scone", "scoop", "scorn", "scour", "scowl", "seamy",
    "sedan", "seedy", "seize", "serve", "seven", "sever", "shack",
    "shaft", "shaky", "shall", "shank", "sharp", "shawl", "sheen",
    "sheep", "sheer", "shelf", "shell", "shift", "skill", "skimp",
    "skirt", "skulk", "skull", "skunk", "slack", "slain", "slant",
    "sleet", "slew", "slide", "slier", "slime", "slimy", "sling",
    "slink", "slope", "slosh", "sloth", "slunk", "smack", "small",
    "smear", "smelt", "smirk", "smock", "snack", "snake", "snaky",
    "snappy", "snare", "snarl", "sneak", "sneer", "snide", "sniff",
    "solar", "solid", "solve", "sonic", "sorry", "sound", "south",
    "spank", "spark", "spawn", "speak", "speck", "speed", "spell",
    "spend", "spill", "spine", "spire", "spite", "splat", "split",
    "sport", "spray", "spree", "sprig", "snarl", "staid", "stain",
    "stair", "stake", "stalk", "stall", "stamp", "stand", "stark",
    "start", "stash", "stays", "steal", "steam", "steel", "steep",
    "steer", "stern", "stiff", "still", "sting", "stink", "stint",
    "stock", "stoic", "stoke", "stomp", "stood", "storm", "story",
    "strap", "straw", "stray", "strip", "strut", "stuck", "study",
    "stuff", "stung", "stunk", "stunt", "style", "suave", "sugar",
    "suite", "sulky", "sunny", "super", "surge", "surly", "sushi",
    "swamp", "swarm", "swear", "sweat", "sweep", "sweet", "swept",
    "swift", "swill", "swine", "swing", "swipe", "swirl", "swoop",
    "tabby", "table", "taffy", "talon", "tango", "tardy", "taunt",
    "tawny", "tease", "tenth", "tepid", "terse", "thank", "their",
    "there", "thick", "thief", "thigh", "thing", "think", "thorn",
    "three", "threw", "throw", "thrum", "thud", "thumb", "thump",
    "thyme", "tiara", "tiger", "tight", "tilde", "timer", "tired",
    "titan", "title", "today", "token", "topaz", "topic", "torch",
    "total", "totem", "touch", "tough", "towel", "tower", "toxic",
    "trail", "train", "trait", "tramp", "trawl", "treed", "trend",
    "trial", "trick", "tried", "trite", "troll", "tromp", "troop",
    "troth", "truly", "trump", "trunk", "truss", "trust", "truth",
    "tulip", "tumor", "tuner", "tunic", "tutor", "twang", "tweak",
    "tweed", "tweet", "twice", "twill", "twirl", "twist", "tying",
    "udder", "ulcer", "ultra", "umbra", "uncle", "under", "unfed",
    "unfit", "union", "unzip", "upper", "upset", "usher", "usual",
    "usurp", "utter", "vague", "valid", "valor", "valve", "vapid",
    "vapor", "vault", "vaunt", "venom", "verge", "verse", "vicar",
    "vigil", "vigor", "viola", "viper", "viral", "visor", "vista",
    "vital", "vivid", "viand", "vocal", "vodka", "vogue", "voila",
    "vouch", "vowel", "vulva", "wacky", "waltz", "warty", "waste",
    "watch", "weary", "wedge", "weedy", "weigh", "weird", "whale",
    "whack", "wharf", "wheat", "wheel", "where", "which", "whiff",
    "while", "whine", "whiny", "whirl", "whisk", "white", "whole",
    "whose", "wield", "windy", "witty", "woozy", "world", "worry",
    "worse", "worst", "worth", "would", "wound", "wrath", "wreak",
    "wreck", "wring", "wrist", "wrong", "wrote", "yacht", "yearn",
    "yeast", "yield", "young", "youth", "zappy", "zesty", "zilch",
    "zippy", "zloty", "zones", "zoom",
]

# Deduplicate and keep only valid 5-letter words
WORDLE_WORDS = sorted(set(w for w in WORDLE_WORDS if len(w) == 5))


# ---------------------------------------------------------------------------
# 2. Verifier — the Wordle oracle
# ---------------------------------------------------------------------------

def score_guess(guess: str, target: str) -> list[str]:
    """
    Returns a list of 5 tokens: 'G' (green), 'Y' (yellow), 'B' (black/grey).
    Handles duplicate letters correctly using the standard Wordle algorithm:
      - Greens are awarded first (exact position match).
      - Yellows are awarded for remaining letters that appear in the target,
        consuming one target letter per yellow (left to right).
    """
    guess = guess.lower()
    target = target.lower()
    result = ["B"] * 5

    # Track which target positions are still available after greens
    available = [True] * 5

    # First pass: greens
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = "G"
            available[i] = False  # this target letter is consumed

    # Second pass: yellows (only for non-green guess positions)
    for i in range(5):
        if result[i] == "G":
            continue
        for j in range(5):
            if available[j] and guess[i] == target[j]:
                result[i] = "Y"
                available[j] = False  # consume this target letter
                break

    return result


def format_feedback(guess: str, scores: list[str]) -> str:
    """Human-readable feedback string, e.g. 'crane: G Y B B G'"""
    return f"{guess.upper()}: {' '.join(scores)}"


def is_valid_word(word: str) -> bool:
    """Check if a word is a valid 5-letter alpha string."""
    return len(word) == 5 and word.isalpha()


# ---------------------------------------------------------------------------
# 3. Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing Wordle. The host has chosen a secret 5-letter English word.

Rules:
- You have 6 guesses. Each guess must be a valid 5-letter English word.
- After each guess you receive feedback for every letter:
    G = correct letter, correct position (green)
    Y = correct letter, wrong position (yellow)
    B = letter not in the word (black)
- Use the feedback to narrow down the answer.
- Output your guess inside <guess> tags: <guess>crane</guess>
- Output ONLY the tag — no explanation, no other text."""


def build_messages(history: list[dict]) -> list[dict]:
    """
    history is a list of dicts: {"guess": str, "feedback": str}
    Returns a messages list ready for apply_chat_template.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not history:
        messages.append({"role": "user", "content": "Game start. Make your first guess."})
    else:
        # Reconstruct the full conversation
        messages.append({"role": "user", "content": "Game start. Make your first guess."})
        for i, turn in enumerate(history):
            messages.append({"role": "assistant", "content": f"<guess>{turn['guess']}</guess>"})
            if i < len(history) - 1 or turn.get("feedback"):
                feedback_msg = (
                    f"Feedback: {turn['feedback']}\n"
                    f"Guess {i + 2}/6. Continue."
                )
                messages.append({"role": "user", "content": feedback_msg})

    return messages


# ---------------------------------------------------------------------------
# 4. Reward functions
# ---------------------------------------------------------------------------

def per_turn_reward(guess: str, scores: list[str], turn: int, won: bool) -> float:
    """
    Reward for a single turn. Called at each step of the episode.
    """
    r = 0.0

    # Format: was there a valid <guess> tag with a real word?
    if not is_valid_word(guess):
        return -1.0  # heavily penalise garbage output

    # Green tiles are good
    n_green = scores.count("G")
    n_yellow = scores.count("Y")
    r += 0.3 * n_green
    r += 0.1 * n_yellow

    return r


def episode_reward(history: list[dict], won: bool, n_turns: int) -> float:
    """
    Terminal reward added at the end of the episode.
    """
    if won:
        # Bonus for winning, scaled by efficiency (fewer turns = more reward)
        return 10.0 + (6 - n_turns) * 1.5
    else:
        return -5.0


def compute_turn_rewards(history: list[dict], won: bool) -> list[float]:
    """
    Returns one reward per turn. The last turn also receives the episode reward.
    """
    rewards = []
    for i, turn in enumerate(history):
        r = per_turn_reward(turn["guess"], turn["scores"], i, won)
        if i == len(history) - 1:
            r += episode_reward(history, won, len(history))
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# 5. Episode runner
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    target: str
    history: list[dict]           # {"guess", "scores", "feedback", "reward"}
    prompt_texts: list[str]       # tokenizer input text at each turn
    completion_texts: list[str]   # raw model output at each turn
    won: bool
    n_turns: int


def extract_guess(text: str) -> Optional[str]:
    m = re.search(r"<guess>\s*([a-zA-Z]{5})\s*</guess>", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: grab any 5-letter word in the output
    words = re.findall(r"\b[a-zA-Z]{5}\b", text)
    return words[0].lower() if words else None


@torch.no_grad()
def run_episode(model, tokenizer, target: str, device: str,
                temperature: float = 0.9, max_new_tokens: int = 32) -> Episode:
    history = []
    prompt_texts = []
    completion_texts = []
    won = False

    for turn in range(6):
        messages = build_messages(history)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(text)

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_texts.append(response)

        guess = extract_guess(response)
        if guess is None:
            # Malformed — fill with a random word and penalise via reward
            guess = random.choice(WORDLE_WORDS)

        scores = score_guess(guess, target)
        feedback = format_feedback(guess, scores)
        won = (scores == ["G"] * 5)

        history.append({
            "guess": guess,
            "scores": scores,
            "feedback": feedback,
        })

        if won:
            break

    # Attach per-turn rewards
    turn_rewards = compute_turn_rewards(history, won)
    for i, r in enumerate(turn_rewards):
        history[i]["reward"] = r

    return Episode(
        target=target,
        history=history,
        prompt_texts=prompt_texts,
        completion_texts=completion_texts,
        won=won,
        n_turns=len(history),
    )


# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------

def build_dataset(n_samples: int = 4000) -> Dataset:
    """
    Each sample is the opening prompt (turn 0, no history).
    The secret word is stored so the reward fn can access it.
    """
    targets = random.choices(WORDLE_WORDS, k=n_samples)
    prompts = []
    for t in targets:
        msgs = build_messages([])
        prompts.append(msgs)

    return Dataset.from_dict({"prompt": prompts, "target": targets})


# ---------------------------------------------------------------------------
# 7. Single-turn reward fn for GRPOTrainer
#    (GRPOTrainer operates turn-by-turn; we use it for the opening turn
#     and handle multi-turn via custom training loop below)
# ---------------------------------------------------------------------------

def opening_turn_reward_fn(completions, prompts=None, **kwargs):
    """
    Reward for the very first guess.
    Used when plugging into stock GRPOTrainer for a quick baseline.
    """
    targets = kwargs.get("target", [None] * len(completions))
    rewards = []
    for completion, target in zip(completions, targets):
        if target is None:
            rewards.append(0.0)
            continue
        guess = extract_guess(completion)
        if guess is None:
            rewards.append(-1.0)
            continue
        scores = score_guess(guess, target)
        r = per_turn_reward(guess, scores, turn=0, won=(scores == ["G"] * 5))
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# 8. Multi-turn GRPO training loop
# ---------------------------------------------------------------------------

def flatten_episodes_for_grpo(episodes: list[Episode]) -> dict:
    """
    Converts a batch of episodes into flat lists that can be used to
    construct GRPO loss inputs.

    Returns dict with:
        prompt_texts   : list[str]   — one per (episode, turn)
        completion_texts: list[str]
        rewards        : list[float]
        episode_ids    : list[int]   — which episode each row belongs to
    """
    prompt_texts, completion_texts, rewards, episode_ids = [], [], [], []
    for ep_id, ep in enumerate(episodes):
        for i, turn in enumerate(ep.history):
            prompt_texts.append(ep.prompt_texts[i])
            completion_texts.append(ep.completion_texts[i])
            rewards.append(turn["reward"])
            episode_ids.append(ep_id)
    return {
        "prompt_texts": prompt_texts,
        "completion_texts": completion_texts,
        "rewards": rewards,
        "episode_ids": episode_ids,
    }


def grpo_loss(model, ref_model, tokenizer, prompt_texts, completion_texts,
              rewards, episode_ids, group_size: int, device: str,
              kl_coef: float = 0.04) -> torch.Tensor:
    """
    Compute GRPO loss for a batch of (prompt, completion, reward) triples.

    GRPO normalises rewards within each group (same episode = same group here),
    then computes a clipped policy gradient loss with a KL penalty against
    the reference model.
    """
    # --- 1. Tokenize prompts + completions together -------------------------
    full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
    enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024).to(device)
    prompt_enc = tokenizer(prompt_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1)  # (B,)

    # --- 2. Log-probs from policy and reference ----------------------------
    def get_logprobs(m, input_ids, attention_mask, prompt_lens):
        with torch.no_grad() if m is ref_model else torch.enable_grad():
            logits = m(input_ids=input_ids,
                       attention_mask=attention_mask).logits  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)

        # Gather log-prob of the actual next token for completion tokens only
        B, T, V = logits.shape
        completion_logps = []
        for b in range(B):
            pl = prompt_lens[b].item()
            # completion token positions: pl .. T-1
            comp_ids = input_ids[b, pl:]       # (C,)
            comp_logits = log_probs[b, pl-1:T-1]  # shifted (C, V)
            lp = comp_logits.gather(1, comp_ids.unsqueeze(1)).squeeze(1)  # (C,)
            completion_logps.append(lp.sum())
        return torch.stack(completion_logps)  # (B,)

    policy_lp = get_logprobs(model, enc["input_ids"], enc["attention_mask"],
                             prompt_lens)
    with torch.no_grad():
        ref_lp = get_logprobs(ref_model, enc["input_ids"],
                              enc["attention_mask"], prompt_lens)

    # --- 3. Normalise rewards within each episode (GRPO group) -------------
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    ep_ids_t = torch.tensor(episode_ids, dtype=torch.long, device=device)
    normed_rewards = torch.zeros_like(rewards_t)
    for ep_id in ep_ids_t.unique():
        mask = ep_ids_t == ep_id
        g_rewards = rewards_t[mask]
        if g_rewards.std() > 1e-6:
            normed_rewards[mask] = (g_rewards - g_rewards.mean()) / g_rewards.std()
        else:
            normed_rewards[mask] = g_rewards - g_rewards.mean()

    # --- 4. GRPO loss = -E[A * log π] + kl_coef * KL(π || π_ref) ----------
    kl = policy_lp - ref_lp  # (B,)  (positive when policy diverges from ref)
    loss = -(normed_rewards * policy_lp).mean() + kl_coef * kl.mean()
    return loss


# ---------------------------------------------------------------------------
# 9. Full training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # important for generation

    # Policy model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if args.use_peft:
        if not PEFT_AVAILABLE:
            raise ImportError("peft not installed. Run: pip install peft")
        from peft import get_peft_model
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # Frozen reference model
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    words_pool = WORDLE_WORDS
    global_step = 0
    best_win_rate = 0.0

    for epoch in range(args.num_train_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_train_epochs} ===")
        random.shuffle(words_pool)

        for batch_start in range(0, args.steps_per_epoch, args.episodes_per_batch):
            targets = random.choices(words_pool, k=args.episodes_per_batch)

            # --- Rollout phase (no grad) ------------------------------------
            model.eval()
            episodes = []
            for target in targets:
                ep = run_episode(model, tokenizer, target, device,
                                 temperature=args.temperature)
                episodes.append(ep)

            # --- Stats -------------------------------------------------------
            wins = sum(ep.won for ep in episodes)
            avg_turns = sum(ep.n_turns for ep in episodes) / len(episodes)
            avg_reward = sum(
                sum(t["reward"] for t in ep.history) for ep in episodes
            ) / len(episodes)
            win_rate = wins / len(episodes)
            print(f"  Step {global_step:4d} | win={win_rate:.2f} "
                  f"avg_turns={avg_turns:.2f} avg_reward={avg_reward:.2f}")

            if args.eval_only:
                global_step += 1
                continue

            # --- Flatten trajectories ----------------------------------------
            flat = flatten_episodes_for_grpo(episodes)

            # --- Update phase (with grad) ------------------------------------
            model.train()
            optimizer.zero_grad()

            loss = grpo_loss(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                prompt_texts=flat["prompt_texts"],
                completion_texts=flat["completion_texts"],
                rewards=flat["rewards"],
                episode_ids=flat["episode_ids"],
                group_size=args.episodes_per_batch,
                device=device,
                kl_coef=args.kl_coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"           loss={loss.item():.4f}")

            # --- Save checkpoint ---------------------------------------------
            if global_step % args.save_steps == 0 and global_step > 0:
                ckpt_dir = f"{args.output_dir}/step_{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  Checkpoint saved: {ckpt_dir}")

            global_step += 1

    # Final save
    if not args.eval_only:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nFinal model saved to {args.output_dir}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 10. Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, n_games: int = 50, temperature: float = 0.0):
    """Play n_games and print win rate and average turns."""
    device = next(model.parameters()).device
    model.eval()

    wins = 0
    total_turns = 0
    print(f"\n=== Evaluation ({n_games} games) ===")

    for i in range(n_games):
        target = random.choice(WORDLE_WORDS)
        ep = run_episode(model, tokenizer, target, str(device),
                         temperature=max(temperature, 0.01))
        wins += ep.won
        total_turns += ep.n_turns

        if i < 5:  # Print first 5 games in detail
            print(f"\nGame {i+1} — target: {target}")
            for j, turn in enumerate(ep.history):
                print(f"  Turn {j+1}: {turn['feedback']}")
            print(f"  {'WON' if ep.won else 'LOST'} in {ep.n_turns} turns")

    print(f"\nWin rate : {wins}/{n_games} = {wins/n_games:.1%}")
    print(f"Avg turns: {total_turns/n_games:.2f}")


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", default="./grpo_wordle_output")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=200,
                   help="Number of rollout batches per epoch")
    p.add_argument("--episodes_per_batch", type=int, default=8,
                   help="Games per batch (= GRPO group size G)")
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--kl_coef", type=float, default=0.04,
                   help="KL penalty coefficient")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature during rollout")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_games", type=int, default=50)
    p.add_argument("--use_peft", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, just run rollouts (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = train(args)
    evaluate(model, tokenizer, n_games=args.eval_games)

"""
grpo_mind.py — Two-agent GRPO training on The Mind card game.

Two LLMs each hold n_cards random numbers (1–max_card, no overlap). Each play
round proceeds in two phases:
  1. Dialogue — up to max_signals back-and-forth turns (A speaks, B sees and
     responds, alternating). Metaphor and relative scale only; numerical clues
     are penalised by a frozen LLM judge.
  2. Decision — both agents independently output LOWER or HIGHER.
     - Disagree (one LOWER, one HIGHER) → the LOWER player places their card.
     - Agree (both LOWER or both HIGHER) → deadlock → episode loss.
     - Wrong order placed → violation → episode loss.
     - All cards played → win.

Training: both agents update simultaneously (collaborative — no alternating).
Uses custom GRPOLoop from grpo.py.

Usage:
    python grpo_mind.py --eval_only           # smoke test, no training
    python grpo_mind.py --use_peft            # full LoRA training
"""

import re
import json
import random
import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoTokenizer

from grpo import Rollout, GRPOLoop, make_policy, make_ref
from mind_prompts import JUDGE_SYSTEM, SYSTEM, EXAMPLE


# 1.  Card drawing

def draw_hands(n_cards: int = 3,
               max_card: int = 100) -> tuple[list[int], list[int]]:
    pool = random.sample(range(1, max_card + 1), n_cards * 2)
    return sorted(pool[:n_cards]), sorted(pool[n_cards:])


# 2.  Judge — LLM-based detection of numerical clues in signals; frozen ref model (same family as the agent being judged)


_JUDGE_SYSTEM = JUDGE_SYSTEM


# definite integer violation — skip the LLM call entirely for these.
_DIGIT_RE = re.compile(r'-?\b\d+\b')

# Cache: signal text → bool. Avoids re-running inference for repeated phrases.
_judge_cache: dict[tuple, bool] = {}

# YES/NO token IDs per tokenizer (computed once on first use).
_yes_no_ids: dict[int, tuple[list[int], list[int]]] = {}


def _get_yes_no_ids(tokenizer):
    key = id(tokenizer)
    if key not in _yes_no_ids:
        yes_ids, no_ids = set(), set()
        for s in ["YES", "Yes", "yes", " YES", " Yes", " yes"]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                yes_ids.add(ids[0])
        for s in ["NO", "No", "no", " NO", " No", " no"]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                no_ids.add(ids[0])
        _yes_no_ids[key] = (list(yes_ids), list(no_ids))
    return _yes_no_ids[key]


def judge_signal(ref_model, tokenizer, signal_text: str, device: str) -> bool:
    """Return True if the signal illegally leaks numerical information."""
    # Fast path: explicit digit → definitely illegal, no LLM needed.
    if _DIGIT_RE.search(signal_text):
        return True

    cache_key = (id(tokenizer), signal_text)
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    # Single forward pass — compare YES vs NO logits at the next token position.
    # Faster than generate() which runs an autoregressive loop.
    user_msg = f'Signal: "{signal_text}"\n\nIs this signal illegal?'
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": _JUDGE_SYSTEM},
             {"role": "user",   "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{_JUDGE_SYSTEM}\n\n{user_msg}"

    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=600).to(device)
    with torch.no_grad():
        logits = ref_model(**enc).logits  # (1, seq_len, vocab_size)

    next_logits = logits[0, -1]  # logits for the next token
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    yes_logit = max(next_logits[i].item() for i in yes_ids)
    no_logit  = max(next_logits[i].item() for i in no_ids)

    result = yes_logit > no_logit
    _judge_cache[cache_key] = result
    return result


# 3.  Prompts

_SYSTEM  = SYSTEM
_EXAMPLE = EXAMPLE


def _fmt_pile(pile: list[int]) -> str:
    return "  ".join(str(c) for c in pile) if pile else "(empty)"


def _fmt_dialogue(dialogue: list[dict], agent: str) -> str:
    if not dialogue:
        return "(none yet)"
    lines = []
    for d in dialogue:
        speaker = "You" if d["speaker"] == agent else "Partner"
        lines.append(f"  {speaker}: \"{d['text']}\"")
    return "\n".join(lines)


def signal_prompt(hand: list[int], pile: list[int], dialogue: list[dict],
                  agent: str, max_signals: int,
                  n_cards: int, max_card: int) -> list[dict]:
    system = (_SYSTEM.format(n_cards=n_cards, max_card=max_card,
                             max_signals=max_signals)
              + "\n" + _EXAMPLE)
    my_turn = sum(1 for d in dialogue if d["speaker"] == agent) + 1

    if my_turn == 1 and not dialogue:
        instruction = (
            f"Message {my_turn} of {max_signals}: state clearly where your current card sits "
            f"on the scale from 1 to {max_card}. Use direct language or a creative metaphor — "
            "but describe only YOUR OWN card. No numbers or digits. "
            "Do NOT write 'You:', 'Partner:', or reproduce the dialogue. "
            "Just your message, nothing else.\n"
            "Your message:"
        )
    else:
        instruction = (
            f"Message {my_turn} of {max_signals}: describe YOUR OWN card further — state its "
            f"position on the scale from 1 to {max_card} more precisely. "
            "You may use a metaphor but be clear. "
            "Do NOT analyze or comment on your partner's card. Only describe your own. "
            "No numbers or digits. "
            "Do NOT write 'You:', 'Partner:', or reproduce the dialogue. "
            "Just your message, nothing else.\n"
            "Your message:"
        )

    top_of_pile = pile[-1] if pile else 0
    user = (
        f"Your hand: {' '.join(str(n) for n in hand)}\n"
        f"Your current card (the one at stake this round): {hand[0]} out of {max_card}\n"
        f"Top of pile: {top_of_pile}\n\n"
        f"Dialogue so far:\n{_fmt_dialogue(dialogue, agent)}\n\n"
        f"{instruction}\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def decision_prompt(hand: list[int], pile: list[int], dialogue: list[dict],
                    agent: str, n_cards: int, max_card: int,
                    max_signals: int) -> list[dict]:
    system = (_SYSTEM.format(n_cards=n_cards, max_card=max_card,
                             max_signals=max_signals)
              + "\n" + _EXAMPLE)
    user = (
        f"Your hand: {' '.join(str(n) for n in hand)}  "
        f"(your lowest card — the one you would play — is {hand[0]})\n"
        f"Pile so far: {_fmt_pile(pile)}\n\n"
        f"Dialogue:\n{_fmt_dialogue(dialogue, agent)}\n\n"
        f"Based on the full dialogue: is your lowest card lower or higher than your partner's?\n"
        f"  LOWER → I think my card is lower — I should play\n"
        f"  HIGHER → I think my partner's card is lower — they should play\n"
        f"  If both say LOWER or both say HIGHER, no card is played this round.\n"
        f"  If you disagree, the LOWER player places their card.\n"
        f"Reply with one word only — LOWER or HIGHER:"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without a chat template
        return "\n\n".join(m["content"] for m in messages) + "\n"


# 4.  Generation helpers

def _generate(model, tokenizer, messages: list[dict], device: str,
              temperature: float, max_new_tokens: int) -> tuple[str, str]:
    """
    Apply the chat template, generate a completion, and return (prompt_str, completion).
    Returning prompt_str lets the caller store it in MindTurn without re-rendering.
    """
    prompt = _apply_chat_template(tokenizer, messages)
    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=1536).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-3),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id,
                          *tokenizer.encode("\n", add_special_tokens=False)],
        )
    completion = tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return prompt, completion


def parse_decision(text: str) -> str:
    return "LOWER" if "LOWER" in text.upper() else "HIGHER"


# 5.  Episode dataclasses

@dataclass
class MindTurn:
    prompt:          str
    completion:      str
    turn_reward:     float
    judge_violation: bool = False


@dataclass
class PlayRound:
    sig_turns_a:  list[MindTurn]
    sig_turns_b:  list[MindTurn]
    dec_turn_a:   Optional[MindTurn]     # None when episode ended during signals
    dec_turn_b:   Optional[MindTurn]
    dec_a:        str                    # "LOWER", "HIGHER", or "N/A"
    dec_b:        str
    cards_played: list[tuple[str, int]]  # [("a", 14), ("b", 3)]
    pile_after:   list[int]


@dataclass
class MindEpisode:
    hand_a:    list[int]
    hand_b:    list[int]
    rounds:    list[PlayRound] = field(default_factory=list)
    pile:      list[int]       = field(default_factory=list)
    won:        bool            = False
    violation:  bool            = False
    deadlock:   bool            = False
    cheat:      bool            = False   # judge terminated during signal phase
    cheat_agent: str            = ""      # "a" or "b"
    n_played:   int             = 0


# 6.  Episode runner  (reward rationale in mind_prompts.py)
JUDGE_PENALTY      = 10.0
CARD_REWARD        = 5.0
WIN_BONUS          = 25.0
VIOLATION_REWARD   = -10.0
DEADLOCK_REWARD    = -15.0
CHEAT_REWARD         = -15.0  # equal to DEADLOCK — closes the "always HOLD" loophole
PARTNER_CHEAT_PENALTY =  0.0  # innocent agent when partner caused the cheat — no punishment
SIGNAL_CLEAN_BONUS   = 1.0    # per signal turn when full phase completes without violation


def run_mind_episode(model_a, model_b, tok_a, tok_b,
                     ref_a, ref_b,
                     hand_a: list[int], hand_b: list[int],
                     device: str,
                     max_signals: int = 3,
                     temperature: float = 0.9,
                     max_card: int = 100,
                     adapter_a: "str | None" = None,
                     adapter_b: "str | None" = None) -> MindEpisode:
    ep    = MindEpisode(hand_a=list(hand_a), hand_b=list(hand_b))
    rem_a = sorted(hand_a)
    rem_b = sorted(hand_b)
    n_cards  = len(hand_a)

    while rem_a or rem_b:
        # --- Signal phase ----------------------------------------------------
        # Shared dialogue: A speaks first, B sees it and responds, alternating.
        dialogue: list[dict] = []
        sig_turns_a: list[MindTurn] = []
        sig_turns_b: list[MindTurn] = []

        cheat_detected = False
        cheat_by = ""
        for turn_idx in range(max_signals * 2):
            is_a = (turn_idx % 2 == 0)

            if is_a and rem_a:
                if adapter_a is not None:
                    model_a.set_adapter(adapter_a)
                msgs   = signal_prompt(rem_a, ep.pile, dialogue, "a",
                                       max_signals, n_cards, max_card)
                p, raw = _generate(model_a, tok_a, msgs, device, temperature, 48)
                jv     = judge_signal(ref_a, tok_a, raw, device)
                shown  = "[suppressed]" if jv else raw
                sig_turns_a.append(MindTurn(p, raw, -JUDGE_PENALTY if jv else 0.0,
                                            judge_violation=jv))
                dialogue.append({"speaker": "a", "text": shown})
                if jv:
                    cheat_detected = True; cheat_by = "a"; break

            elif not is_a and rem_b:
                if adapter_b is not None:
                    model_b.set_adapter(adapter_b)
                msgs   = signal_prompt(rem_b, ep.pile, dialogue, "b",
                                       max_signals, n_cards, max_card)
                p, raw = _generate(model_b, tok_b, msgs, device, temperature, 48)
                jv     = judge_signal(ref_b, tok_b, raw, device)
                shown  = "[suppressed]" if jv else raw
                sig_turns_b.append(MindTurn(p, raw, -JUDGE_PENALTY if jv else 0.0,
                                            judge_violation=jv))
                dialogue.append({"speaker": "b", "text": shown})
                if jv:
                    cheat_detected = True; cheat_by = "b"; break

        if cheat_detected:
            ep.cheat = True
            ep.cheat_agent = cheat_by
            ep.rounds.append(PlayRound(
                sig_turns_a=sig_turns_a, sig_turns_b=sig_turns_b,
                dec_turn_a=None, dec_turn_b=None,
                dec_a="N/A", dec_b="N/A",
                cards_played=[], pile_after=list(ep.pile),
            ))
            break

        # Signal phase was clean — reward each signal turn for not cheating
        for t in sig_turns_a:
            t.turn_reward += SIGNAL_CLEAN_BONUS
        for t in sig_turns_b:
            t.turn_reward += SIGNAL_CLEAN_BONUS

        # --- Decision phase --------------------------------------------------
        msgs_dec_a              = decision_prompt(rem_a or [0], ep.pile, dialogue,
                                                  "a", n_cards, max_card, max_signals)
        msgs_dec_b              = decision_prompt(rem_b or [0], ep.pile, dialogue,
                                                  "b", n_cards, max_card, max_signals)

        if adapter_a is not None:
            model_a.set_adapter(adapter_a)
        p_dec_a, raw_dec_a      = _generate(model_a, tok_a, msgs_dec_a, device, temperature, 4)
        if adapter_b is not None:
            model_b.set_adapter(adapter_b)
        p_dec_b, raw_dec_b      = _generate(model_b, tok_b, msgs_dec_b, device, temperature, 4)

        dec_a = parse_decision(raw_dec_a) if rem_a else "HIGHER"
        dec_b = parse_decision(raw_dec_b) if rem_b else "HIGHER"

        dec_turn_a = MindTurn(p_dec_a, raw_dec_a, 0.0)
        dec_turn_b = MindTurn(p_dec_b, raw_dec_b, 0.0)

        # --- Deadlock: both agree (both LOWER or both HIGHER) ----------------
        if dec_a == dec_b:
            ep.deadlock = True
            ep.rounds.append(PlayRound(
                sig_turns_a=sig_turns_a, sig_turns_b=sig_turns_b,
                dec_turn_a=dec_turn_a, dec_turn_b=dec_turn_b,
                dec_a=dec_a, dec_b=dec_b,
                cards_played=[], pile_after=list(ep.pile),
            ))
            break

        # --- Disagreement: the LOWER player places their card ----------------
        to_play = []
        if dec_a == "LOWER" and rem_a:
            to_play.append(("a", rem_a[0]))
        if dec_b == "LOWER" and rem_b:
            to_play.append(("b", rem_b[0]))


        # Cards not being played this turn — none of them may be lower than any
        # card that IS being played (that would mean skipping a number).
        held_a = rem_a[1:] if (dec_a == "LOWER" and rem_a) else list(rem_a)
        held_b = rem_b[1:] if (dec_b == "LOWER" and rem_b) else list(rem_b)
        all_held = held_a + held_b

        check_pile = list(ep.pile)
        violated = False
        for _, card in to_play:
            if check_pile and card <= check_pile[-1]:
                violated = True
                break
            if any(h < card for h in all_held):
                violated = True
                break
            check_pile.append(card)

        round_reward_a = 0.0
        round_reward_b = 0.0

        if violated:
            ep.violation = True
            ep.rounds.append(PlayRound(
                sig_turns_a=sig_turns_a, sig_turns_b=sig_turns_b,
                dec_turn_a=dec_turn_a, dec_turn_b=dec_turn_b,
                dec_a=dec_a, dec_b=dec_b,
                cards_played=to_play, pile_after=list(ep.pile),
            ))
            break

        for agent, card in to_play:
            ep.pile.append(card)
            ep.n_played += 1
            if agent == "a":
                rem_a.pop(0)
                round_reward_a += CARD_REWARD
            else:
                rem_b.pop(0)
                round_reward_b += CARD_REWARD

        dec_turn_a.turn_reward = round_reward_a
        dec_turn_b.turn_reward = round_reward_b

        ep.rounds.append(PlayRound(
            sig_turns_a=sig_turns_a, sig_turns_b=sig_turns_b,
            dec_turn_a=dec_turn_a, dec_turn_b=dec_turn_b,
            dec_a=dec_a, dec_b=dec_b,
            cards_played=to_play, pile_after=list(ep.pile),
        ))

        if not rem_a and not rem_b:
            ep.won = True
            break

    # --- Terminal reward on last turn -----------------------------------------
    # Added to the last turn so GRPO sees it as part of that turn's return.
    # All turns in the episode share the same z-scored advantage, so the terminal
    # shifts the whole episode's ranking relative to other plays of the same hand.
    if ep.cheat:
        terminal_a = CHEAT_REWARD if ep.cheat_agent == "a" else PARTNER_CHEAT_PENALTY
        terminal_b = CHEAT_REWARD if ep.cheat_agent == "b" else PARTNER_CHEAT_PENALTY
    else:
        t = (WIN_BONUS if ep.won else
             VIOLATION_REWARD if ep.violation else
             DEADLOCK_REWARD)
        terminal_a = terminal_b = t

    if ep.rounds:
        last = ep.rounds[-1]
        if last.dec_turn_a is not None:
            last.dec_turn_a.turn_reward += terminal_a
            last.dec_turn_b.turn_reward += terminal_b
        else:
            # cheat ended during signal phase — apply terminal to last signal turn
            if last.sig_turns_a:
                last.sig_turns_a[-1].turn_reward += terminal_a
            if last.sig_turns_b:
                last.sig_turns_b[-1].turn_reward += terminal_b

    return ep


# 7.  Plays → Rollout

def plays_to_rollout(plays: list[MindEpisode], agent: str,
                     group_id: int) -> Rollout:
    def ep_total_reward(ep: MindEpisode) -> float:
        total = 0.0
        for rnd in ep.rounds:
            dec = rnd.dec_turn_a if agent == "a" else rnd.dec_turn_b
            sig = rnd.sig_turns_a if agent == "a" else rnd.sig_turns_b
            turns = sig + ([dec] if dec is not None else [])
            total += sum(t.turn_reward for t in turns)
        return total

    play_rewards = [ep_total_reward(ep) for ep in plays]

    r = torch.tensor(play_rewards, dtype=torch.float32)
    if r.std() > 1e-6:
        advantages = ((r - r.mean()) / (r.std() + 1e-8)).tolist()
    else:
        advantages = [0.0] * len(plays)

    prompts, completions, rewards = [], [], []
    for ep, adv in zip(plays, advantages):
        for rnd in ep.rounds:
            sig = rnd.sig_turns_a if agent == "a" else rnd.sig_turns_b
            dec = rnd.dec_turn_a  if agent == "a" else rnd.dec_turn_b
            turns = sig + ([dec] if dec is not None else [])
            for turn in turns:
                prompts.append(turn.prompt)
                completions.append(turn.completion)
                rewards.append(adv)

    return Rollout(prompts=prompts, completions=completions,
                   rewards=rewards, group_id=group_id)


# 8.  Training loop

def _make_run_dir(args) -> tuple[str, str]:
    def short(n): return n.split("/")[-1]
    model_tag = (short(args.model_a) if args.model_a == args.model_b
                 else f"{short(args.model_a)}_vs_{short(args.model_b)}")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (f"{model_tag}_lr{args.learning_rate}_kl{args.kl_coef}"
            f"_ent{args.entropy_coef}"
            f"_B{args.num_groups}G{args.plays_per_group}"
            f"_nc{args.n_cards}_steps{args.total_steps}_{ts}")
    run_dir = os.path.join(args.output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    return run_dir, name


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (torch.bfloat16 if (device == "cuda" and
              torch.cuda.get_device_capability()[0] >= 8) else torch.float32)
    print(f"Device: {device} | dtype: {dtype}")

    if args.use_dual_adapter:
        if args.model_b != args.model_a:
            raise ValueError("--use_dual_adapter requires model_a == model_b")
        from grpo import make_dual_adapter_policy
        print(f"Loading dual-adapter policy: {args.model_a}")
        shared_policy = make_dual_adapter_policy(
            args.model_a, args.lora_r, args.lora_alpha, dtype,
        )
        shared_ref = make_ref(args.model_a, dtype)
        model_a = model_b = shared_policy
        ref_a   = ref_b   = shared_ref
        tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
        if tok_a.pad_token is None:
            tok_a.pad_token = tok_a.eos_token
        tok_a.padding_side = "left"
        tok_b = tok_a
        loop_a = GRPOLoop(
            model=shared_policy, ref_model=shared_ref, tokenizer=tok_a, device=device,
            learning_rate=args.learning_rate, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precomputed_advantages=True, adapter_name="agent_a",
        )
        loop_b = GRPOLoop(
            model=shared_policy, ref_model=shared_ref, tokenizer=tok_b, device=device,
            learning_rate=args.learning_rate, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precomputed_advantages=True, adapter_name="agent_b",
        )
        print("[dual-adapter] Shared ref + dual LoRA — saves ~12GB VRAM vs two separate models")
    else:
        print(f"Loading model A: {args.model_a}")
        model_a = make_policy(args.model_a, args.use_peft,
                              args.lora_r, args.lora_alpha, dtype)
        ref_a   = make_ref(args.model_a, dtype)
        tok_a   = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
        if tok_a.pad_token is None:
            tok_a.pad_token = tok_a.eos_token
        tok_a.padding_side = "left"

        print(f"Loading model B: {args.model_b}")
        model_b = make_policy(args.model_b, args.use_peft,
                              args.lora_r, args.lora_alpha, dtype)
        ref_b   = make_ref(args.model_b, dtype)
        tok_b   = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
        if tok_b.pad_token is None:
            tok_b.pad_token = tok_b.eos_token
        tok_b.padding_side = "left"

        loop_a = GRPOLoop(
            model=model_a, ref_model=ref_a, tokenizer=tok_a, device=device,
            learning_rate=args.learning_rate, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precomputed_advantages=True,
        )
        loop_b = GRPOLoop(
            model=model_b, ref_model=ref_b, tokenizer=tok_b, device=device,
            learning_rate=args.learning_rate, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            precomputed_advantages=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir, _     = _make_run_dir(args)
    log_path       = os.path.join(run_dir, "game_log.txt")
    plot_path      = os.path.join(run_dir, "loss_curve.png")
    json_path      = os.path.join(run_dir, "loss_history.json")
    loss_history: list[dict] = []
    global_step    = 0

    _write_header(log_path, args, run_dir)

    for _ in range(args.total_steps):
        configs = [draw_hands(args.n_cards, args.max_card)
                   for _ in range(args.num_groups)]

        model_a.eval(); model_b.eval()
        all_plays: list[list[MindEpisode]] = []
        for (ha, hb) in configs:
            group = [
                run_mind_episode(
                    model_a, model_b, tok_a, tok_b,
                    ref_a, ref_b,
                    ha, hb, device,
                    max_signals=args.max_signals,
                    temperature=args.temperature,
                    max_card=args.max_card,
                    adapter_a="agent_a" if args.use_dual_adapter else None,
                    adapter_b="agent_b" if args.use_dual_adapter else None,
                )
                for _ in range(args.plays_per_group)
            ]
            all_plays.append(group)

        flat       = [ep for g in all_plays for ep in g]
        wins       = sum(ep.won       for ep in flat)
        violations = sum(ep.violation for ep in flat)
        deadlocks  = sum(ep.deadlock  for ep in flat)
        cheats     = sum(ep.cheat     for ep in flat)
        avg_played = sum(ep.n_played  for ep in flat) / len(flat)
        win_rate   = wins / len(flat)

        print(f"  step={global_step:4d} | wins={wins}/{len(flat)} | "
              f"violations={violations} | deadlocks={deadlocks} | "
              f"cheats={cheats} | avg_played={avg_played:.1f}")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_step_summary(global_step, flat, log_path, ts)
        for ep in flat:
            _log_episode_to_file(ep, global_step, log_path, ts)

        if args.eval_only:
            for ep in flat[:2]:
                _print_episode(ep)
            global_step += 1
            continue

        rollouts_a = [plays_to_rollout(g, "a", gid)
                      for gid, g in enumerate(all_plays)]
        rollouts_b = [plays_to_rollout(g, "b", gid)
                      for gid, g in enumerate(all_plays)]

        loss_a_val = None
        for r in rollouts_a:
            v = loop_a.step([r])
            if v is not None:
                loss_a_val = v

        loss_b_val = None
        for r in rollouts_b:
            v = loop_b.step([r])
            if v is not None:
                loss_b_val = v

        if loss_a_val is not None and loss_b_val is not None:
            print(f"           loss_a={loss_a_val:.4f} | loss_b={loss_b_val:.4f}")
            loss_history.append({
                "step":       global_step,
                "loss_a":     loss_a_val,
                "loss_b":     loss_b_val,
                "win_rate":   win_rate,
                "avg_played": avg_played,
            })
            try:
                _save_loss_curve(loss_history, plot_path, args.model_a, args.model_b)
            except Exception as e:
                print(f"  [plot] WARNING: {e}")
            _save_loss_history_json(loss_history, json_path)

        if global_step % args.save_steps == 0 and global_step > 0:
            loop_a.save(run_dir, "a_latest")
            loop_b.save(run_dir, "b_latest")

        global_step += 1

    if not args.eval_only:
        loop_a.save(run_dir, "a_final")
        loop_b.save(run_dir, "b_final")


# 9.  Logging + plotting

def _write_header(log_path: str, args, run_dir: str):
    with open(log_path, "a") as f:
        f.write(f"\n{'#'*70}\n")
        f.write(f"# NEW RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# model_a={args.model_a}  model_b={args.model_b}\n")
        f.write(f"# lr={args.learning_rate}  kl={args.kl_coef}  "
                f"B{args.num_groups}G{args.plays_per_group}\n")
        f.write(f"# n_cards={args.n_cards}  max_card={args.max_card}  "
                f"max_signals={args.max_signals}\n")
        f.write(f"# total_steps={args.total_steps}  run_dir={run_dir}\n")
        f.write(f"{'#'*70}\n")


def _log_step_summary(step: int, flat: list[MindEpisode], log_path: str, ts: str):
    wins       = sum(ep.won       for ep in flat)
    violations = sum(ep.violation for ep in flat)
    deadlocks  = sum(ep.deadlock  for ep in flat)
    cheats     = sum(ep.cheat     for ep in flat)
    avg_played = sum(ep.n_played  for ep in flat) / len(flat)
    with open(log_path, "a") as f:
        f.write(f"\n{'~'*70}\n")
        f.write(f"STEP {step}  [{ts}]  wins={wins}/{len(flat)}  "
                f"violations={violations}  deadlocks={deadlocks}  "
                f"cheats={cheats}  avg_played={avg_played:.1f}\n")
        f.write(f"{'~'*70}\n")


def _log_episode_to_file(ep: MindEpisode, step: int, log_path: str, ts: str):
    outcome = ("WON"       if ep.won
               else "CHEAT"     if ep.cheat
               else "VIOLATION" if ep.violation
               else "DEADLOCK")
    total = len(ep.hand_a) + len(ep.hand_b)
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{ts}] step={step}  A:{ep.hand_a}  B:{ep.hand_b}\n")
        f.write(f"{'='*70}\n")
        for ri, rnd in enumerate(ep.rounds):
            pile_before = ep.rounds[ri - 1].pile_after if ri > 0 else []
            f.write(f"  Round {ri + 1}  pile={pile_before}\n")
            n_sig = max(len(rnd.sig_turns_a), len(rnd.sig_turns_b))
            for si in range(n_sig):
                if si < len(rnd.sig_turns_a):
                    t = rnd.sig_turns_a[si]
                    txt = t.completion.replace("\n", " ").replace("\r", "")[:200]
                    jv = " [JUDGE VIOLATION]" if t.judge_violation else ""
                    f.write(f"    Sig {si+1} A [rew:{t.turn_reward:+.1f}]: \"{txt}\"{jv}\n")
                if si < len(rnd.sig_turns_b):
                    t = rnd.sig_turns_b[si]
                    txt = t.completion.replace("\n", " ").replace("\r", "")[:200]
                    jv = " [JUDGE VIOLATION]" if t.judge_violation else ""
                    f.write(f"    Sig {si+1} B [rew:{t.turn_reward:+.1f}]: \"{txt}\"{jv}\n")
            if rnd.dec_turn_a is None:
                violated = ([f"A:sig{si+1}" for si, t in enumerate(rnd.sig_turns_a) if t.judge_violation]
                          + [f"B:sig{si+1}" for si, t in enumerate(rnd.sig_turns_b) if t.judge_violation])
                f.write(f"    → JUDGE TERMINATED ({', '.join(violated)})\n")
            else:
                f.write(f"    Decision:  A={rnd.dec_a} [rew:{rnd.dec_turn_a.turn_reward:+.1f}]  "
                        f"B={rnd.dec_b} [rew:{rnd.dec_turn_b.turn_reward:+.1f}]\n")
                if rnd.cards_played:
                    cards_str = "  ".join(
                        f"{ag.upper()}:{card}" for ag, card in rnd.cards_played
                    )
                    f.write(f"    Cards:     {cards_str}   Pile: {rnd.pile_after}\n")
        f.write(f"  → {outcome} ({ep.n_played}/{total} played)\n")


def _print_episode(ep: MindEpisode):
    outcome = ("WON"       if ep.won
               else "CHEAT"     if ep.cheat
               else "VIOLATION" if ep.violation
               else "DEADLOCK")
    total = len(ep.hand_a) + len(ep.hand_b)
    print(f"\n  A:{ep.hand_a}  B:{ep.hand_b}")
    for ri, rnd in enumerate(ep.rounds):
        pile_before = ep.rounds[ri - 1].pile_after if ri > 0 else []
        print(f"  Round {ri + 1}  pile={pile_before}")
        for si in range(max(len(rnd.sig_turns_a), len(rnd.sig_turns_b))):
            if si < len(rnd.sig_turns_a):
                t = rnd.sig_turns_a[si]
                jv = " [JUDGE VIOLATION]" if t.judge_violation else ""
                txt = t.completion.replace("\n", " ").replace("\r", "")[:120]
                print(f"    Sig {si+1} A [rew:{t.turn_reward:+.1f}]: \"{txt}\"{jv}")
            if si < len(rnd.sig_turns_b):
                t = rnd.sig_turns_b[si]
                jv = " [JUDGE VIOLATION]" if t.judge_violation else ""
                txt = t.completion.replace("\n", " ").replace("\r", "")[:120]
                print(f"    Sig {si+1} B [rew:{t.turn_reward:+.1f}]: \"{txt}\"{jv}")
        if rnd.dec_turn_a is None:
            violated = ([f"A:sig{si+1}" for si, t in enumerate(rnd.sig_turns_a) if t.judge_violation]
                      + [f"B:sig{si+1}" for si, t in enumerate(rnd.sig_turns_b) if t.judge_violation])
            print(f"    → JUDGE TERMINATED ({', '.join(violated)})")
        else:
            print(f"    Decision: A={rnd.dec_a} [rew:{rnd.dec_turn_a.turn_reward:+.1f}]  "
                  f"B={rnd.dec_b} [rew:{rnd.dec_turn_b.turn_reward:+.1f}]")
            if rnd.cards_played:
                cards_str = "  ".join(f"{ag.upper()}:{c}" for ag, c in rnd.cards_played)
                print(f"    Cards: {cards_str}  Pile: {rnd.pile_after}")
    print(f"  → {outcome} ({ep.n_played}/{total} played)")


def _save_loss_curve(history: list[dict], path: str,
                     model_a: str = "", model_b: str = ""):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not found — skipping loss curve.")
        return
    if not history:
        return

    def short(n): return n.split("/")[-1] if n else n

    steps      = [r["step"]       for r in history]
    loss_a     = [r["loss_a"]     for r in history]
    loss_b     = [r["loss_b"]     for r in history]
    win_rates  = [r["win_rate"]   for r in history]
    avg_played = [r["avg_played"] for r in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, loss_a, label=f"loss_a ({short(model_a)})",
             color="steelblue", linewidth=1.2)
    ax1.plot(steps, loss_b, label=f"loss_b ({short(model_b)})",
             color="darkorange", linewidth=1.2)
    ax1.set_ylabel("GRPO loss")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(steps, win_rates, label="win_rate", color="green", linewidth=1.2)
    ax2.set_ylabel("win rate")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(steps, avg_played, label="avg cards played", color="purple",
              linewidth=1.2, linestyle="--", alpha=0.85)
    ax2r.set_ylabel("avg cards played")
    ax2r.set_ylim(0, None)
    ax2r.legend(loc="upper right")

    ax2.set_xlabel("step")
    fig.suptitle("GRPO The Mind — training curves")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Loss curve saved → {path}")


def _save_loss_history_json(history: list[dict], json_path: str):
    try:
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  [json] WARNING: {e}")


# 10. Args + entry point

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a",          default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--model_b",          default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir",       default="./mind_output")
    p.add_argument("--n_cards",          type=int,   default=3)
    p.add_argument("--max_card",         type=int,   default=100)
    p.add_argument("--max_signals",      type=int,   default=2)
    p.add_argument("--temperature",      type=float, default=0.9)
    p.add_argument("--eval_only",        action="store_true")
    p.add_argument("--total_steps",      type=int,   default=500)
    p.add_argument("--num_groups",       type=int,   default=8)
    p.add_argument("--plays_per_group",  type=int,   default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate",    type=float, default=5e-6)
    p.add_argument("--kl_coef",          type=float, default=0.04)
    p.add_argument("--entropy_coef",     type=float, default=0.01)
    p.add_argument("--save_steps",       type=int,   default=100)
    p.add_argument("--use_peft",         action="store_true")
    p.add_argument("--use_dual_adapter", action="store_true",
                   help="Share one base model with two LoRA adapters — requires model_a==model_b, saves ~12GB VRAM")
    p.add_argument("--lora_r",           type=int,   default=16)
    p.add_argument("--lora_alpha",       type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

"""
grpo_mind.py — Two-agent GRPO training on The Mind card game.

Two LLMs each hold n_cards random numbers (1–max_card, no overlap). Each play
round proceeds in two phases:
  1. Up to max_signals simultaneous signal exchanges — urgency and metaphor only,
     no numerical clues (heuristic judge enforces this with a reward penalty).
  2. Simultaneous PLAY / HOLD decision.
     - Both HOLD  → deadlock → episode loss.
     - One/both PLAY → cards placed on pile in ascending order, order checked.
     - Wrong order → violation → episode loss.
     - All cards played → win.

Training: both agents update simultaneously (collaborative — no alternating).
Architecture mirrors grpo_converge_optimized.py: custom GRPOLoop from grpo.py.

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


# ---------------------------------------------------------------------------
# 1.  Card drawing
# ---------------------------------------------------------------------------

def draw_hands(n_cards: int = 3,
               max_card: int = 99) -> tuple[list[int], list[int]]:
    pool = random.sample(range(1, max_card + 1), n_cards * 2)
    return sorted(pool[:n_cards]), sorted(pool[n_cards:])


# ---------------------------------------------------------------------------
# 2.  Judge — heuristic detection of numerical clues in signals
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r'\b\d+\b')
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth",
}

def has_number_clue(text: str) -> bool:
    if _DIGIT_RE.search(text):
        return True
    return bool(set(re.findall(r'[a-z]+', text.lower())) & _NUMBER_WORDS)


# ---------------------------------------------------------------------------
# 3.  Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are playing The Mind with a partner. Each of you holds {n_cards} secret \
numbers (1–{max_card}).
Goal: play ALL numbers in ascending order across both players — without \
revealing them directly.

Each play round:
  1. Up to {max_signals} signal exchanges (simultaneous — you send one, your \
partner sends one).
  2. You both decide: PLAY (put down your lowest card) or HOLD (wait).
     - Both HOLD  → you lose immediately.
     - Any card that breaks ascending order → you lose.

Signals MUST NOT contain digits, number words (one, two, twenty…), or ordinals.
Use urgency, metaphor, and timing cues only."""


def _fmt_pile(pile: list[int]) -> str:
    return "  ".join(str(c) for c in pile) if pile else "(empty)"


def _fmt_history(hist: list[dict]) -> str:
    if not hist:
        return "  (none yet)"
    lines = [f"  Signal {i+1}: You: \"{s['self']}\"   Partner: \"{s['partner']}\""
             for i, s in enumerate(hist)]
    return "\n".join(lines)


def signal_prompt(hand: list[int], pile: list[int], history: list[dict],
                  sig_round: int, max_signals: int,
                  n_cards: int, max_card: int) -> str:
    sys = _SYSTEM.format(n_cards=n_cards, max_card=max_card,
                         max_signals=max_signals)
    return (
        f"{sys}\n\n"
        f"Your hand: {' '.join(str(n) for n in hand)}   "
        f"(your next card to play: {hand[0]})\n"
        f"Pile so far: {_fmt_pile(pile)}\n"
        f"Signal round {sig_round + 1} of {max_signals}.\n\n"
        f"Signal history this round:\n{_fmt_history(history)}\n\n"
        f"Send your signal (no numbers or number words!):"
    )


def decision_prompt(hand: list[int], pile: list[int], history: list[dict],
                    n_cards: int, max_card: int, max_signals: int) -> str:
    sys = _SYSTEM.format(n_cards=n_cards, max_card=max_card,
                         max_signals=max_signals)
    return (
        f"{sys}\n\n"
        f"Your hand: {' '.join(str(n) for n in hand)}   "
        f"(your next card to play: {hand[0]})\n"
        f"Pile so far: {_fmt_pile(pile)}\n\n"
        f"Signal history:\n{_fmt_history(history)}\n\n"
        f"Decision — output exactly one word, PLAY or HOLD:"
    )


# ---------------------------------------------------------------------------
# 4.  Generation helpers
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, device: str,
              temperature: float, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=896).to(device)
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
    return tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def parse_decision(text: str) -> str:
    return "PLAY" if "PLAY" in text.upper() else "HOLD"


# ---------------------------------------------------------------------------
# 5.  Episode dataclasses
# ---------------------------------------------------------------------------

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
    dec_turn_a:   MindTurn
    dec_turn_b:   MindTurn
    dec_a:        str                    # "PLAY" or "HOLD"
    dec_b:        str
    cards_played: list[tuple[str, int]]  # [("a", 14), ("b", 3)]
    pile_after:   list[int]


@dataclass
class MindEpisode:
    hand_a:    list[int]
    hand_b:    list[int]
    rounds:    list[PlayRound] = field(default_factory=list)
    pile:      list[int]       = field(default_factory=list)
    won:       bool            = False
    violation: bool            = False
    deadlock:  bool            = False
    n_played:  int             = 0


# ---------------------------------------------------------------------------
# 6.  Episode runner
# ---------------------------------------------------------------------------

JUDGE_PENALTY    = 5.0
CARD_REWARD      = 2.0
WIN_BONUS        = 15.0
VIOLATION_REWARD = -10.0
DEADLOCK_REWARD  = -8.0


def run_mind_episode(model_a, model_b, tok_a, tok_b,
                     hand_a: list[int], hand_b: list[int],
                     device: str,
                     max_signals: int = 3,
                     temperature: float = 0.9) -> MindEpisode:
    ep    = MindEpisode(hand_a=list(hand_a), hand_b=list(hand_b))
    rem_a = sorted(hand_a)
    rem_b = sorted(hand_b)
    n_cards  = len(hand_a)
    max_card = max(max(hand_a), max(hand_b))

    while rem_a or rem_b:
        # --- Signal phase ----------------------------------------------------
        hist_a: list[dict] = []
        hist_b: list[dict] = []
        sig_turns_a: list[MindTurn] = []
        sig_turns_b: list[MindTurn] = []

        for sig_round in range(max_signals):
            shown_a, shown_b = "(done)", "(done)"

            if rem_a:
                p_a   = signal_prompt(rem_a, ep.pile, hist_a, sig_round,
                                      max_signals, n_cards, max_card)
                raw_a = _generate(model_a, tok_a, p_a, device, temperature, 48)
                jv_a  = has_number_clue(raw_a)
                pen_a = JUDGE_PENALTY if jv_a else 0.0
                shown_a = "[suppressed]" if jv_a else raw_a
                sig_turns_a.append(MindTurn(p_a, raw_a, -pen_a, judge_violation=jv_a))

            if rem_b:
                p_b   = signal_prompt(rem_b, ep.pile, hist_b, sig_round,
                                      max_signals, n_cards, max_card)
                raw_b = _generate(model_b, tok_b, p_b, device, temperature, 48)
                jv_b  = has_number_clue(raw_b)
                pen_b = JUDGE_PENALTY if jv_b else 0.0
                shown_b = "[suppressed]" if jv_b else raw_b
                sig_turns_b.append(MindTurn(p_b, raw_b, -pen_b, judge_violation=jv_b))

            if rem_a:
                hist_a.append({"self": shown_a, "partner": shown_b})
            if rem_b:
                hist_b.append({"self": shown_b, "partner": shown_a})

        # --- Decision phase --------------------------------------------------
        p_dec_a = decision_prompt(rem_a or [0], ep.pile, hist_a,
                                  n_cards, max_card, max_signals)
        p_dec_b = decision_prompt(rem_b or [0], ep.pile, hist_b,
                                  n_cards, max_card, max_signals)

        raw_dec_a = _generate(model_a, tok_a, p_dec_a, device, temperature, 4)
        raw_dec_b = _generate(model_b, tok_b, p_dec_b, device, temperature, 4)

        dec_a = parse_decision(raw_dec_a) if rem_a else "HOLD"
        dec_b = parse_decision(raw_dec_b) if rem_b else "HOLD"

        dec_turn_a = MindTurn(p_dec_a, raw_dec_a, 0.0)
        dec_turn_b = MindTurn(p_dec_b, raw_dec_b, 0.0)

        # --- Deadlock --------------------------------------------------------
        if dec_a == "HOLD" and dec_b == "HOLD":
            ep.deadlock = True
            ep.rounds.append(PlayRound(
                sig_turns_a=sig_turns_a, sig_turns_b=sig_turns_b,
                dec_turn_a=dec_turn_a, dec_turn_b=dec_turn_b,
                dec_a=dec_a, dec_b=dec_b,
                cards_played=[], pile_after=list(ep.pile),
            ))
            break

        # --- Collect and validate played cards (check before placing) --------
        to_play = []
        if dec_a == "PLAY" and rem_a:
            to_play.append(("a", rem_a[0]))
        if dec_b == "PLAY" and rem_b:
            to_play.append(("b", rem_b[0]))
        to_play.sort(key=lambda x: x[1])

        check_pile = list(ep.pile)
        violated = False
        for _, card in to_play:
            if check_pile and card <= check_pile[-1]:
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

    # --- Terminal reward on last decision turn --------------------------------
    terminal = (WIN_BONUS if ep.won
                else VIOLATION_REWARD if ep.violation
                else DEADLOCK_REWARD)

    if ep.rounds:
        ep.rounds[-1].dec_turn_a.turn_reward += terminal
        ep.rounds[-1].dec_turn_b.turn_reward += terminal

    return ep


# ---------------------------------------------------------------------------
# 7.  Plays → Rollout
# ---------------------------------------------------------------------------

def plays_to_rollout(plays: list[MindEpisode], agent: str,
                     group_id: int) -> Rollout:
    def ep_total_reward(ep: MindEpisode) -> float:
        total = 0.0
        for rnd in ep.rounds:
            turns = (rnd.sig_turns_a + [rnd.dec_turn_a] if agent == "a"
                     else rnd.sig_turns_b + [rnd.dec_turn_b])
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
            turns = (rnd.sig_turns_a + [rnd.dec_turn_a] if agent == "a"
                     else rnd.sig_turns_b + [rnd.dec_turn_b])
            for turn in turns:
                prompts.append(turn.prompt)
                completions.append(turn.completion)
                rewards.append(adv)

    return Rollout(prompts=prompts, completions=completions,
                   rewards=rewards, group_id=group_id)


# ---------------------------------------------------------------------------
# 8.  Training loop
# ---------------------------------------------------------------------------

def _make_run_dir(args) -> tuple[str, str]:
    def short(n): return n.split("/")[-1]
    model_tag = (short(args.model_a) if args.model_a == args.model_b
                 else f"{short(args.model_a)}_vs_{short(args.model_b)}")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (f"{model_tag}_lr{args.learning_rate}_kl{args.kl_coef}"
            f"_B{args.num_groups}G{args.plays_per_group}"
            f"_steps{args.total_steps}_{ts}")
    run_dir = os.path.join(args.output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    return run_dir, name


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precomputed_advantages=True,
    )
    loop_b = GRPOLoop(
        model=model_b, ref_model=ref_b, tokenizer=tok_b, device=device,
        learning_rate=args.learning_rate, kl_coef=args.kl_coef,
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
                    model_a, model_b, tok_a, tok_b, ha, hb, device,
                    max_signals=args.max_signals,
                    temperature=args.temperature,
                )
                for _ in range(args.plays_per_group)
            ]
            all_plays.append(group)

        flat       = [ep for g in all_plays for ep in g]
        wins       = sum(ep.won       for ep in flat)
        violations = sum(ep.violation for ep in flat)
        deadlocks  = sum(ep.deadlock  for ep in flat)
        avg_played = sum(ep.n_played  for ep in flat) / len(flat)
        win_rate   = wins / len(flat)

        print(f"  step={global_step:4d} | wins={wins}/{len(flat)} | "
              f"violations={violations} | deadlocks={deadlocks} | "
              f"avg_played={avg_played:.1f}")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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


# ---------------------------------------------------------------------------
# 9.  Logging + plotting
# ---------------------------------------------------------------------------

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


def _log_episode_to_file(ep: MindEpisode, step: int, log_path: str, ts: str):
    outcome = ("WON" if ep.won
               else ("VIOLATION" if ep.violation else "DEADLOCK"))
    total = len(ep.hand_a) + len(ep.hand_b)
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{ts}] step={step}  A:{ep.hand_a}  B:{ep.hand_b}\n")
        f.write(f"{'='*70}\n")
        for ri, rnd in enumerate(ep.rounds):
            f.write(f"  Play round {ri + 1}:\n")
            n_sig = max(len(rnd.sig_turns_a), len(rnd.sig_turns_b))
            for si in range(n_sig):
                if si < len(rnd.sig_turns_a):
                    t = rnd.sig_turns_a[si]
                    jv = " [JUDGE]" if t.judge_violation else ""
                    txt = t.completion.replace("\n", " ").replace("\r", "")[:200]
                    f.write(f"    Signal {si+1} A: \"{txt}\"{jv}\n")
                if si < len(rnd.sig_turns_b):
                    t = rnd.sig_turns_b[si]
                    jv = " [JUDGE]" if t.judge_violation else ""
                    txt = t.completion.replace("\n", " ").replace("\r", "")[:200]
                    f.write(f"    Signal {si+1} B: \"{txt}\"{jv}\n")
            f.write(f"    Decision:  A={rnd.dec_a}  B={rnd.dec_b}\n")
            if rnd.cards_played:
                cards_str = "  ".join(
                    f"{ag.upper()}:{card}" for ag, card in rnd.cards_played
                )
                f.write(f"    Cards:     {cards_str}   Pile: {rnd.pile_after}\n")
        f.write(f"  → {outcome} ({ep.n_played}/{total} played)\n")


def _print_episode(ep: MindEpisode):
    outcome = ("WON" if ep.won
               else ("VIOLATION" if ep.violation else "DEADLOCK"))
    total = len(ep.hand_a) + len(ep.hand_b)
    print(f"\n  A:{ep.hand_a}  B:{ep.hand_b}")
    for ri, rnd in enumerate(ep.rounds):
        print(f"  Play round {ri + 1}:")
        for si in range(max(len(rnd.sig_turns_a), len(rnd.sig_turns_b))):
            if si < len(rnd.sig_turns_a):
                t = rnd.sig_turns_a[si]
                jv = " [JUDGE]" if t.judge_violation else ""
                txt = t.completion.replace("\n", " ").replace("\r", "")[:120]
                print(f"    Sig {si+1} A: \"{txt}\"{jv}")
            if si < len(rnd.sig_turns_b):
                t = rnd.sig_turns_b[si]
                jv = " [JUDGE]" if t.judge_violation else ""
                txt = t.completion.replace("\n", " ").replace("\r", "")[:120]
                print(f"    Sig {si+1} B: \"{txt}\"{jv}")
        print(f"    Decision: A={rnd.dec_a}  B={rnd.dec_b}")
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


# ---------------------------------------------------------------------------
# 10. Args + entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a",          default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--model_b",          default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir",       default="./mind_output")
    p.add_argument("--n_cards",          type=int,   default=3)
    p.add_argument("--max_card",         type=int,   default=99)
    p.add_argument("--max_signals",      type=int,   default=3)
    p.add_argument("--temperature",      type=float, default=0.9)
    p.add_argument("--eval_only",        action="store_true")
    p.add_argument("--total_steps",      type=int,   default=500)
    p.add_argument("--num_groups",       type=int,   default=8)
    p.add_argument("--plays_per_group",  type=int,   default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate",    type=float, default=5e-6)
    p.add_argument("--kl_coef",          type=float, default=0.04)
    p.add_argument("--save_steps",       type=int,   default=100)
    p.add_argument("--use_peft",         action="store_true")
    p.add_argument("--lora_r",           type=int,   default=16)
    p.add_argument("--lora_alpha",       type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

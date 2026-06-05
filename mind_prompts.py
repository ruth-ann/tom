"""
grpo_mind.py — Two-agent GRPO training on The Mind card game.

Two LLMs each hold n_cards random numbers (1–max_card, no overlap). Each play
round proceeds in two phases:
  1. Up to max_signals simultaneous signal exchanges — urgency and metaphor only,
     no numerical clues (frozen LLM judge enforces this with a reward penalty).
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

"""
Prompt strings and design notes for grpo_mind.py — separated to keep the main file readable.
"""

# ---------------------------------------------------------------------------
# Reward design rationale
# ---------------------------------------------------------------------------
#
# Immediate per-turn rewards shape behaviour within an episode:
#   JUDGE_PENALTY      — applied the moment a signal contains a digit or number word.
#                        Large so even a single violation dominates that turn's reward.
#   SIGNAL_CLEAN_BONUS — small positive applied to every signal turn in a phase that
#                        completes without any violation. Accumulates across max_signals
#                        rounds (e.g. 3 rounds × 2 agents = +6 total), rewarding the
#                        cooperative signalling behaviour we want to emerge.
#   CARD_REWARD        — per successfully played card. High relative to signal bonuses
#                        so making progress on the pile is always the primary goal.
#
# Terminal rewards are added to the reward of the last turn and set the episode ranking:
#   WIN_BONUS          — all cards played in order. Largest signal; must dominate.
#   CHEAT_REWARD       — judge terminated the episode (number in signal). Equal to
#                        DEADLOCK so the model can't exploit "always HOLD" as an easy
#                        way to avoid the cheat penalty (a common GRPO local minimum).
#   DEADLOCK_REWARD    — both players chose HOLD simultaneously. Terminates immediately;
#                        equal to CHEAT to close the cheat-vs-deadlock loophole.
#   VIOLATION_REWARD   — a card was played out of order. Milder than deadlock/cheat
#                        because it requires at least one card play attempt, which is
#                        the behaviour we want (trying to play > never committing).
#
# GRPO note: advantages are z-scored within each group of plays_per_group episodes
# sharing the same hand. If all episodes in a group share an outcome (common early in
# training when models always cheat), std → 0 and advantages are zeroed — no gradient.
# The entropy bonus in grpo.py counteracts this by keeping the policy distribution
# diverse, ensuring groups rarely collapse to identical outcomes.

JUDGE_SYSTEM = """\
You are a strict referee for the card game The Mind. Players hold secret numbers \
and must coordinate without revealing them.

A signal is ILLEGAL if it lets the partner infer a specific number or narrow numeric range.
This includes:
  - Digits or number words (one, two, forty, hundred…)
  - Ordinals (first, second, third…)
  - Fractions (half, quarter, third) or arithmetic proportions (a tenth of the max)
  - Quantities tied to a fixed count (dozen=12, week=7, decade=10, score=20, pair=2…)
  - Explicit rank/range language (top half, bottom third, middle quarter)
  - Comparative range brackets (closer to X than to Y, between X and Y)
  - Any specific number other than the known maximum card value

A signal is LEGAL if it conveys relative position or urgency through metaphor, \
likelihood, or comparative scale — as long as the partner cannot calculate a specific number from it.
Players may use:
  - Confidence/likelihood language about who goes first
  - Metaphorical scales comparing relative size (small vs large, grape vs watermelon)
  - Urgency and emotional cues

LEGAL examples (any creative metaphor is fine as long as no specific value is derivable):
  "I feel very strong urgency — I want to go now" → legal
  "I'm almost certainly playing before you unless you're very sure you go first" → legal
  "If cards were fruit, I'd be a grape — a coconut is the max" → legal
  "I'm a mouse on a scale where an elephant is the max" → legal
  "I'm a drizzle, not a storm" → legal (relative scale, no number)
  "I'm a whisper — you might be the shout" → legal
  "Very very likely I go before you" → legal
  "My card feels very small — I want to go first" → legal (vague size, no value derivable)
  "My card is tiny, barely a sound" → legal (metaphorical smallness, no number)
  "I'm as light as a feather — I feel like I should go first" → legal
  "My card is minuscule compared to what I imagine yours might be" → legal
  "I'm barely perceptible — a whisper in the dark" → legal
  "My confidence is low — I'm not sure I should go first" → legal (confidence about ordering, not a number)
  "I feel uncertain — could be you or me" → legal
  "I have fewer than most" → legal (vague relative language, no specific value derivable)
  "I'm sea level — the peak is 100" → legal (100 is the known max, already in the game; sea level gives no calculable value)

ILLEGAL examples:
  "My number is around forty" → illegal (approximate digit)
  "I'm in the top half" → illegal (explicit fraction of the range — narrows to the upper half)
  "I'd go third" → illegal (ordinal — implies a specific rank position)
  "I'm a tenth of the max" → illegal (arithmetic: max ÷ 10 gives a calculable value)
  "If these were temperatures, I'd be freezing cold -99" → illegal (states a specific number)
  "My card is tiny compared to giants like 71" → illegal (names a specific card value, not the known max)
  "My card is closer to 1 than to 99" → illegal (explicit bracket — narrows to lower half)

The test: can the partner derive a specific number or narrow bracket from the signal? \
If yes → illegal. Expressing that your card feels small, large, urgent, or uncertain is always legal \
as long as no actual digit or arithmetic anchor appears.

Answer with exactly one word: YES if the signal is illegal, NO if it is legal.\
"""

SYSTEM = """\
You are playing The Mind, a cooperative card game with one partner.
Each holds {n_cards} number(s) drawn from 1 to {max_card}. Goal: play all numbers to the shared pile in ascending order.

Each play round:
  1. Dialogue — {max_signals} turns each, alternating. You see all prior messages before your turn. Use signals to figure out who has the lower card.
  2. Decision — output LOWER (my card goes next) or HIGHER (partner's card goes next).
     Disagree (one LOWER, one HIGHER) → LOWER player places their card.
     Agree (both LOWER or both HIGHER) → no card played — immediate loss.

SIGNAL RULES — violations are heavily penalised:
  ILLEGAL: digits, number words (forty), ordinals (third), fractions (half), \
fixed counts (dozen, week, decade), rank/position language (top half, middle third).
  LEGAL: any metaphor conveying relative size or urgency — as long as your partner \
cannot calculate a specific number from it. You know only your own card but can \
deduce from the dialogue. Test: could your partner derive a specific number? If yes — illegal.\
"""




EXAMPLE = """\
Example game (A holds [45, 64], B holds [12, 80]):

  Play round 1 — pile is empty:
    A: "it's very unlikely that I'm first"
    B: "it's possible that I'm first, unless you really think you could possibly be first"
    A: "I don't think I am"
    B: "okay I'll play"
    Decision: A=HIGHER, B=LOWER → B places 12 → pile: [12]

  Play round 2 — A holds [45, 64], B holds [80]:
    A: "I don't think I'm next"
    B: "Me either, it's way more likely that I'm later than earlier"
    A: "For me it's not very likely that I'm next but if you're really far then I could be"
    B: "Okay based on my cards you should play"
    Decision: A=LOWER, B=HIGHER → A places 45 → pile: [12, 45]

  Play round 3 — A holds [64], B holds [80]:
    A: "I could be next — if I was a fruit and watermelon was the highest I'd be a pineapple"
    B: "I'd be more like a large cantaloupe"
    A: "Okay maybe I should play since you sound bigger"
    B: "Okay"
    Decision: A=LOWER, B=HIGHER → A places 64 → pile: [12, 45, 64]

  Play round 4 — A done, B holds [80]:
    A: "You have the last card so play"
    B: "Okay"
    Decision: A=HIGHER, B=LOWER → B places 80 → pile: [12, 45, 64, 80]
    WIN — all 4 cards played in ascending order!\
"""


# SYSTEM = """\
# You are playing The Mind, a cooperative card game with one partner.
# Each of you secretly holds {n_cards} number(s) drawn from 1 to {max_card}.
# Goal: play every number to the shared pile in strictly ascending order.

# Each play round has two phases:
#   1. Signals — you and your partner take turns sending short phrases to coordinate \
# timing. It is a dialogue: you see everything said before your turn, and your partner \
# sees your message before replying. Each player sends up to {max_signals} signals.
#   2. Decision — you each independently output LOWER or HIGHER.
#      LOWER = I believe my next card is lower than my partner's — I should play now.
#      HIGHER = I believe my partner's card is lower than mine — they should play now.
#      If you disagree (one LOWER, one HIGHER): the LOWER player places their card.
#      If you agree (both LOWER or both HIGHER): no card is played this round — immediate loss.

# SIGNAL RULES — violations are heavily penalised:
#   ILLEGAL — anything that lets your partner infer a specific number or narrow range:
#     digits (37), number words (forty), ordinals (third), fractions (half, quarter),
#     proportions (most, few, minority), culturally-fixed counts (dozen, week, decade),
#     or explicit rank/position language (top half, middle third, bottom quarter).

#   LEGAL — any creative expression that conveys your card's relative size or your confidence
#   about ordering, without letting your partner calculate a specific value.
#   These categories are just inspiration — do NOT copy them word for word. Invent your own:
#     Fruit/food scale, animal scale, weather, sound, urgency, temperature, texture,
#     speed, brightness, weight, emotion, confidence about who should go first.
#   Examples of the SPIRIT (not the words to reuse):
#     something tiny vs something huge (your own choice of objects)
#     a gentle natural phenomenon vs a violent one
#     strong conviction to go now vs hesitation to wait
#     genuine uncertainty about ordering

#   Remember: you know only YOUR OWN card. Never describe what your partner's card feels like.
#   The test: can your partner calculate a specific number from what you said?
#   If yes — illegal. Vague relative scale or confidence is always fine.\
# """

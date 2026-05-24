"""
Standalone GRPO training engine — task/game agnostic.

Public API:
  Rollout                     — dataclass: prompts, completions, rewards, group_id
  compute_completion_logprobs — chunked forward pass → summed log-probs
  grpo_loss                   — GRPO policy-gradient loss with KL penalty
  GRPOLoop                    — wraps model + ref + optimizer + grad accumulation
  make_model_and_ref          — load policy (+ optional LoRA) and frozen ref copy

Usage: instantiate one GRPOLoop per model. Each loop is independent — the
two-model coordination in grpo_converge_optimized.py lives outside this module.

Smoke test (trivial single-model dummy rollouts, no game logic required):
    python grpo.py
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """
    One group of (prompt, completion, reward) triples for GRPO.

    prompts[i] / completions[i] / rewards[i] are one generation attempt.
    group_id groups rollouts for within-group reward normalisation (each
    episode or prompt position is its own group).
    Retries should be included as separate entries with the same final reward.
    """
    prompts:     list[str]
    completions: list[str]
    rewards:     list[float]
    group_id:    int


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_completion_logprobs(
    model, tokenizer,
    prompt_texts:     list[str],
    completion_texts: list[str],
    device:           str,
    chunk_size:       int = 4,
) -> torch.Tensor:
    """
    Compute the mean log-prob over completion tokens for each
    (prompt, completion) pair. Returns shape (B,).

    Mean (not sum) keeps the scale constant across completions of different
    lengths, so kl_coef=0.04 is correctly sized relative to the policy
    gradient term regardless of completion token count.

    chunk_size limits peak activation memory — reduce if OOM.
    torch.enable_grad() ensures policy log-probs retain a grad_fn even when
    called from inside a no_grad context (e.g. the rollout phase).
    """
    all_results = []

    for start in range(0, len(prompt_texts), chunk_size):
        p_chunk = prompt_texts[start: start + chunk_size]
        c_chunk = completion_texts[start: start + chunk_size]

        full_texts = [p + c for p, c in zip(p_chunk, c_chunk)]
        enc_full = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=800,
        ).to(device)
        enc_prompt = tokenizer(
            p_chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=800,
        ).to(device)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)   # (chunk,)

        with torch.enable_grad():
            logits = model(
                input_ids=enc_full["input_ids"],
                attention_mask=enc_full["attention_mask"],
            ).logits                                               # (chunk, T, V)
            log_probs = F.log_softmax(logits, dim=-1)             # (chunk, T, V)

            for b in range(logits.shape[0]):
                pl  = prompt_lens[b].item()
                ids = enc_full["input_ids"][b, pl:]               # completion token ids
                lp  = log_probs[b, pl - 1: pl - 1 + len(ids)]    # shifted logits
                gathered = lp.gather(1, ids.unsqueeze(1)).squeeze(1)
                num_completion_tokens = max(len(ids), 1)
                all_results.append(gathered.sum() / num_completion_tokens)

    return torch.stack(all_results)                               # (B,)


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    model, ref_model, tokenizer,
    rollouts: list[Rollout],
    device:   str,
    kl_coef:  float = 0.04,
    precomputed_advantages: bool = False,
) -> torch.Tensor:
    """
    Standard GRPO loss over a list of Rollout objects.

    precomputed_advantages=False (default): rewards are z-score normalised
    within each group_id. Use when rewards are raw scalars.

    precomputed_advantages=True: rewards are treated as already-normalised
    advantages and used directly. Use when the caller has pre-normalised at
    a coarser level (e.g. play-level rather than sample-level) to avoid bias
    from unequal numbers of samples per group member.
    """
    all_prompts, all_completions, all_rewards, all_group_ids = [], [], [], []

    for r in rollouts:
        for p, c, rw in zip(r.prompts, r.completions, r.rewards):
            all_prompts.append(p)
            all_completions.append(c)
            all_rewards.append(rw)
            all_group_ids.append(r.group_id)

    if not all_prompts:
        return torch.tensor(0.0, requires_grad=True)

    rewards_t   = torch.tensor(all_rewards,   dtype=torch.float32, device=device)
    group_ids_t = torch.tensor(all_group_ids, dtype=torch.long,    device=device)

    if precomputed_advantages:
        normed = rewards_t
    else:
        normed = torch.zeros_like(rewards_t)
        for gid in group_ids_t.unique():
            mask = group_ids_t == gid
            g    = rewards_t[mask]
            if g.std() > 1e-6:
                normed[mask] = (g - g.mean()) / (g.std() + 1e-8)
            else:
                normed[mask] = g - g.mean()

    policy_lp = compute_completion_logprobs(
        model, tokenizer, all_prompts, all_completions, device,
    )
    with torch.no_grad():
        ref_lp = compute_completion_logprobs(
            ref_model, tokenizer, all_prompts, all_completions, device,
        )

    kl   = policy_lp - ref_lp
    loss = -(normed * policy_lp).mean() + kl_coef * kl.mean()
    return loss


# ---------------------------------------------------------------------------
# GRPOLoop — model + optimizer + gradient accumulation
# ---------------------------------------------------------------------------

class GRPOLoop:
    """
    Wraps a policy model and its frozen reference for GRPO updates.

    Gradient accumulation: step() accumulates gradients and calls
    optimizer.step() only every `gradient_accumulation_steps` calls.
    Returns the loss value (float) when the optimizer actually steps,
    None on intermediate accumulation steps.

    adapter_name: if set, this loop manages one named LoRA adapter on a shared
    base model. The active adapter is switched before every forward pass so
    gradients flow only through this adapter's parameters. This allows two
    GRPOLoop instances to share one base model with two separate adapters,
    cutting VRAM from 3× to 2× model size (e.g. 32GB instead of 48GB for 8B).
    """

    def __init__(
        self, *,
        model,
        ref_model,
        tokenizer,
        device:                      str,
        learning_rate:               float = 5e-6,
        kl_coef:                     float = 0.04,
        gradient_accumulation_steps: int   = 1,
        max_grad_norm:               float = 1.0,
        use_8bit_adam:               bool  = True,
        adapter_name:                "str | None" = None,
        precomputed_advantages:      bool  = False,
    ):
        self.model        = model
        self.ref_model    = ref_model
        self.tokenizer    = tokenizer
        self.device       = device
        self.kl_coef      = kl_coef
        self.grad_accum   = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._accum_count  = 0
        self.adapter_name  = adapter_name
        self.precomputed_advantages = precomputed_advantages

        # When sharing a base model with named adapters, only optimise this
        # adapter's parameters so the two loops don't interfere with each other.
        # set_adapter() must be called first — PEFT marks non-active adapter
        # params as requires_grad=False, so filtering before switching yields 0.
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            trainable = [p for n, p in model.named_parameters()
                         if adapter_name in n and p.requires_grad]
            print(f"[GRPOLoop/{adapter_name}] {len(trainable)} trainable param tensors")
        else:
            trainable = list(filter(lambda p: p.requires_grad, model.parameters()))

        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable, lr=learning_rate, weight_decay=0.01,
                )
                print("[GRPOLoop] Using 8-bit AdamW")
            except ImportError:
                print("[GRPOLoop] bitsandbytes not found — falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable, lr=learning_rate, weight_decay=0.01,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=learning_rate, weight_decay=0.01,
            )

    def step(self, rollouts: list[Rollout]) -> "float | None":
        """
        Compute loss, backward, and (every grad_accum calls) step optimizer.
        Returns float loss when optimizer stepped, else None.
        """
        if self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)
        self.model.train()

        loss   = grpo_loss(
            self.model, self.ref_model, self.tokenizer,
            rollouts, self.device, self.kl_coef,
            precomputed_advantages=self.precomputed_advantages,
        )
        scaled = loss / self.grad_accum
        scaled.backward()
        self._accum_count += 1

        if self._accum_count % self.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_val = loss.item()
            del loss, scaled
            torch.cuda.empty_cache()
            return loss_val

        del loss, scaled
        return None

    def save(self, output_dir: str, label: str):
        ckpt = os.path.join(output_dir, f"model_{label}")
        if self.adapter_name is not None:
            # Save only this adapter's weights — avoids writing the full base model twice.
            self.model.save_pretrained(ckpt, selected_adapters=[self.adapter_name])
        else:
            self.model.save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        print(f"  Saved model_{label} → {ckpt}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def make_policy(
    model_name: str,
    use_peft:   bool,
    lora_r:     int,
    lora_alpha: int,
    dtype:      torch.dtype,
):
    """
    Load a trainable policy model, optionally wrapped with LoRA.
    Enables gradient checkpointing + enable_input_require_grads() for PEFT.
    """
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    policy.config.use_cache = False

    if use_peft:
        if not PEFT_AVAILABLE:
            raise ImportError("peft not installed — run: pip install peft")
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, cfg)
        policy.print_trainable_parameters()

    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    return policy


def make_ref(model_name: str, dtype: torch.dtype):
    """
    Load a frozen reference model. No LoRA, no gradient checkpointing
    (frozen inputs have requires_grad=False, which causes warnings with checkpointing).
    Can be shared across multiple GRPOLoop instances when they share the same
    base model — saves one full model's worth of VRAM.
    """
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def make_model_and_ref(
    model_name: str,
    use_peft:   bool,
    lora_r:     int,
    lora_alpha: int,
    dtype:      torch.dtype,
):
    """Convenience wrapper: load one policy + one dedicated ref copy."""
    return (
        make_policy(model_name, use_peft, lora_r, lora_alpha, dtype),
        make_ref(model_name, dtype),
    )


def make_dual_adapter_policy(
    model_name:  str,
    lora_r:      int,
    lora_alpha:  int,
    dtype:       torch.dtype,
    adapter_a:   str = "agent_a",
    adapter_b:   str = "agent_b",
):
    """
    Load one base model with two independent named LoRA adapters.

    Both adapters share the frozen base weights but have entirely separate
    trainable parameters. Use with two GRPOLoop instances (one per adapter_name)
    to train two agents while only holding one copy of the base weights in VRAM.

    Memory vs. two separate models:
      Two separate 8B fp16 policies: 2 × 16GB = 32GB
      One dual-adapter 8B fp16 policy: 16GB + ~200MB adapters ≈ 16GB
    Saving: ~16GB — enough to fit two 8B agents + one 8B ref on a 44GB A40.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft not installed — run: pip install peft")

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False

    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, cfg, adapter_name=adapter_a)
    base.add_adapter(adapter_b, cfg)

    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    base.set_adapter(adapter_a)
    print(f"Dual-adapter policy ({adapter_a} / {adapter_b}):")
    base.print_trainable_parameters()
    return base


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== grpo.py smoke test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name} ...")
    policy, ref = make_model_and_ref(
        model_name, use_peft=False, lora_r=16, lora_alpha=32, dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    loop = GRPOLoop(
        model=policy, ref_model=ref, tokenizer=tokenizer, device=device,
        learning_rate=5e-6, kl_coef=0.04, gradient_accumulation_steps=2,
    )

    # Dummy rollouts — reward = word count of completion
    texts   = ["hello world", "cat", "the quick brown fox",
               "sun", "moon river wide", "a", "sky blue deep", "running"]
    prompts = ["Complete: "] * 8
    rollouts = [
        Rollout(
            prompts=prompts[i * 2: i * 2 + 2],
            completions=texts[i * 2: i * 2 + 2],
            rewards=[float(len(t.split())) for t in texts[i * 2: i * 2 + 2]],
            group_id=i,
        )
        for i in range(4)
    ]

    print("Step 1 (accumulate — expect None) ...")
    r1 = loop.step(rollouts)
    assert r1 is None, f"Expected None, got {r1}"
    print(f"  result: {r1}  PASS")

    print("Step 2 (optimizer fires — expect float) ...")
    r2 = loop.step(rollouts)
    assert isinstance(r2, float), f"Expected float, got {r2}"
    print(f"  loss: {r2:.4f}  PASS")

    print("=== smoke test PASSED ===")

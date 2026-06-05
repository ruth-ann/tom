import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

# 1) Model & Tokenizer Setup
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load in fp16 — A40 has 48GB VRAM, 8B model needs ~16GB so it fits without quantization.
# low_cpu_mem_usage=True streams each shard directly to GPU, keeping CPU RAM low.
print("Loading model in fp16...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
# Llama 3.1 Instruct ends chat turns with <|eot_id|> (128009), not <|end_of_text|> (128001).
# Without this, generation never hits EOS and always runs to max_completion_length.
model.generation_config.eos_token_id = [128001, 128009]

# Smoke-test CUDA forward pass before doing anything else.
# If this hangs, the GPU/kernel combo is fundamentally broken for neural net ops.
print("Running CUDA smoke test...")
with torch.no_grad():
    _dummy = model(torch.zeros(1, 4, dtype=torch.long).to(model.device))
print("CUDA smoke test passed.")

# 2) LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3) Dataset
SYSTEM_PROMPT = "You are a helpful assistant. Please keep your responses brief."
WORD_TARGET = 25

train_dataset = Dataset.from_dict({
    "prompt": [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]
        for q in [
            # Science
            "Why is the sky blue?", "What is gravity?", "What causes earthquakes?",
            "What is DNA?", "Why do stars twinkle?", "What causes thunder?",
            "Why does ice float?", "What is a black hole?", "How does the immune system work?",
            "What is photosynthesis?",
            # Technology
            "How do computers work?", "What is the internet?", "How does GPS work?",
            "What is artificial intelligence?", "How does Wi-Fi work?",
            "What is a microchip?", "How do touchscreens work?", "What is cloud computing?",
            # History & society
            "What is democracy?", "What caused World War 2?", "What was the Renaissance?",
            "What caused the Great Depression?", "What was the Cold War?",
            "Why was the Berlin Wall built?", "What is capitalism?",
            # Biology & everyday
            "Why do we dream?", "How does memory work?", "Why do leaves change color?",
            "How do birds navigate?", "Why do cats purr?", "How do vaccines work?",
            "Why do we yawn?", "How does sleep work?",
            # Physics & math
            "What is relativity?", "What is quantum mechanics?",
            "How does nuclear energy work?", "What is entropy?",
            "Why is pi important?", "What is the speed of light?",
            "How do magnets work?",
        ]  # 40 unique questions, no repetition
    ]
})

# 4) Reward Function
def brevity_reward_func(completions, **kwargs):
    """
    Tent reward peaked at WORD_TARGET words:
      - Too short (< WORD_TARGET/2): penalized, linearly rising to 1.0 at WORD_TARGET
      - At WORD_TARGET: 1.0
      - Too long (> WORD_TARGET): falls as WORD_TARGET / word_count
    Prevents the model from gaming the reward with 1-2 word gibberish.
    """
    WORD_MIN = WORD_TARGET // 2  # below this, reward < 1.0 and falls to 0 at 0 words
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)

        word_count = len(content.strip().split())
        if word_count == 0:
            reward = 0.0
        elif word_count <= WORD_TARGET:
            reward = word_count / WORD_TARGET   # rises linearly: 0→0 at 0 words, 1.0 at WORD_TARGET
        else:
            reward = WORD_TARGET / word_count   # falls: 0.5 at 2×target, 0.1 at 10×target

        rewards.append(reward)
    return rewards

# 5) Evaluation Function
def evaluate_model(tag):
    print(f"\n{'='*20} EVALUATION: {tag} {'='*20}")
    questions = [
        "Why is the sky blue?",
        "What is photosynthesis?",
        "How do computers work?",
    ]
    # Gradient checkpointing (enabled by GRPOTrainer) breaks autoregressive generation.
    # Disable it and restore the KV cache temporarily for clean inference.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for q in questions:
        chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.3)
        prompt_len = inputs['input_ids'].shape[1]
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        words = len(response.split())
        print(f"Q: {q}\nA ({words} words): {response}\n")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

# 6) Plot Callback — records loss and completion length, saves plot at end
class PlotCallback(TrainerCallback):
    def __init__(self):
        self.steps, self.losses, self.lengths, self.rewards = [], [], [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(float(logs["loss"]))
            self.lengths.append(float(logs.get("completions/mean_length", float("nan"))))
            self.rewards.append(float(logs.get("reward", float("nan"))))

    def on_train_end(self, args, state, control, **kwargs):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ax1.plot(self.steps, self.losses, marker="o", markersize=3)
        ax1.set_ylabel("Loss")
        ax1.set_title("GRPO Training")
        ax1.grid(True)
        ax2.plot(self.steps, self.lengths, marker="o", markersize=3, color="tab:orange")
        ax2.set_ylabel("Mean completion length (tokens)")
        ax2.grid(True)
        ax3.plot(self.steps, self.rewards, marker="o", markersize=3, color="tab:green")
        ax3.set_ylabel("Mean reward")
        ax3.set_xlabel("Step")
        ax3.grid(True)
        plt.tight_layout()
        path = "/home1/rharmstr/tom/grpo_training.png"
        plt.savefig(path, dpi=120)
        print(f"Plot saved to {path}")

# 8) Eval Callback — fires every epoch
class EvalCallback(TrainerCallback):
    def __init__(self, eval_fn, every_n_epochs=1):
        self.eval_fn = eval_fn
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch % self.every_n_epochs == 0:
            self.eval_fn(f"EPOCH {epoch}")
        return control

# 9) Training Config
training_args = GRPOConfig(
    output_dir="./grpo_brevity",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    beta=0.1,
    num_generations=4,
    generation_batch_size=4,
    optim="adamw_8bit",
    max_completion_length=256,        # brief system prompt keeps responses well under this
    learning_rate=5e-5,
    fp16=True,
    logging_steps=1,
    report_to="none",
    dataloader_num_workers=0,        # only 1 CPU core available
)

# 10) Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=brevity_reward_func,
    train_dataset=train_dataset,
    callbacks=[EvalCallback(evaluate_model, every_n_epochs=1), PlotCallback()],
)

# 11) Run
if __name__ == "__main__":
    evaluate_model("BEFORE GRPO")
    trainer.train()
    evaluate_model("AFTER GRPO")

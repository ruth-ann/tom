from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# 1. Load your dataset

dataset = load_dataset("fka/prompts.chat", split="train").select_columns(["prompt"])

print(len(dataset))
# 2. Define a simple reward function
def reward_func(completions, **kwargs):
    """Example: Reward longer completions"""
    return [float(len(completion)) for completion in completions]


# 3. Configure training
training_args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    bf16=False,
    fp16=True,

)

# 4. Initialize and train
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",  # e.g. "Qwen/Qwen2-0.5B-Instruct"
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)
trainer.train()

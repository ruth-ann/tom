import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

# 1) Model & Tokenizer Setup
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model in 4-bit (NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

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
# System prompt says nothing about brevity — model must discover it via reward
SYSTEM_PROMPT = "You are a helpful assistant."
WORD_TARGET = 10

train_dataset = Dataset.from_dict({
    "prompt": [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]
        for q in [
            "What is photosynthesis?",
            "Why is the sky blue?",
            "How do computers work?",
            "What causes earthquakes?",
            "Why do we dream?",
            "What is gravity?",
            "How does memory work?",
            "What is democracy?",
        ] * 50  # 400 total
    ]
})

# 4) Reward Function
def brevity_reward_func(completions, **kwargs):
    """
    <= WORD_TARGET words  → full reward (1.0)
    Every word over target → -0.05 penalty, floored at 0.0
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)

        word_count = len(content.strip().split())
        if word_count <= WORD_TARGET:
            reward = 1.0
        else:
            reward = max(0.0, 1.0 - (word_count - WORD_TARGET) * 0.05)

        rewards.append(reward)
    return rewards

# 5) Evaluation Function
def evaluate_model(tag):
    print(f"\n{'='*20} EVALUATION: {tag} {'='*20}")
    model.eval()
    questions = [
        "Why is the sky blue?",
        "What is photosynthesis?",
        "How do computers work?",
    ]
    for q in questions:
        chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        prompt_len = inputs['input_ids'].shape[1]
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        words = len(response.split())
        print(f"Q: {q}\nA ({words} words): {response}\n")

# 6) Eval Callback — fires every epoch
class EvalCallback(TrainerCallback):
    def __init__(self, eval_fn, every_n_epochs=1):
        self.eval_fn = eval_fn
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch % self.every_n_epochs == 0:
            self.eval_fn(f"EPOCH {epoch}")
        return control

# 7) Training Config
training_args = GRPOConfig(
    output_dir="./grpo_brevity",
    num_train_epochs=3,             # brevity is learned fast, 3 is plenty
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    beta=0.0,                       # no KL penalty = no reference model on GPU
    num_generations=4,
    generation_batch_size=4,
    optim="adamw_8bit",
    max_completion_length=256,       # long enough for model to fail and learn
    learning_rate=1e-4,
    bf16=True,
    logging_steps=5,
    report_to="none",
)

# 8) Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=brevity_reward_func,
    train_dataset=train_dataset,
    callbacks=[EvalCallback(evaluate_model, every_n_epochs=1)],
)

# 9) Run
if __name__ == "__main__":
    evaluate_model("BEFORE GRPO")   # expect: verbose 40-100 word answers
    trainer.train()
    evaluate_model("AFTER GRPO")    # expect: terse <=10 word answers

# import os
# # Must be set BEFORE torch is imported
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
# from trl import GRPOTrainer, GRPOConfig

# # 1) Model & Tokenizer Setup
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# print("Loading model in bfloat16...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# # Enable gradient checkpointing to save activation memory
# model.gradient_checkpointing_enable()

# # 2) Dataset
# SYSTEM_PROMPT = "Respond to the user. Every response must start with the exact string 'begin:'."

# train_dataset = Dataset.from_dict({
#     "prompt": [
#         [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": f"Research Task {i}: Provide a brief statement."}
#         ]
#         for i in range(400)
#     ]
# })

# # 3) Reward Function
# TARGET = "begin:"

# def research_reward_func(completions, **kwargs):
#     rewards = []
#     for completion in completions:
#         if isinstance(completion, list) and len(completion) > 0:
#             content = completion[0].get("content", "")
#         else:
#             content = str(completion)
#         text_clean = content.strip().lower()
        
#         match_count = 0
#         for a, b in zip(text_clean, TARGET):
#             if a == b: match_count += 1
#             else: break
        
#         score = match_count / len(TARGET)
#         if text_clean.startswith(TARGET):
#             score += 2.0
#         rewards.append(score)
#     return rewards

# # 4) Training Configuration (MAX VRAM SAVINGS)
# training_args = GRPOConfig(
#     output_dir="./grpo_results",
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=16, # Increased to maintain effective batch size
    
#     # --- VRAM SAVERS ---
#     num_generations=4,              # Reduced to 4 to save KV cache space
#     generation_batch_size=4,   
#     optim="adamw_8bit",             # Uses bitsandbytes
    
#     max_completion_length=32,       # Shortened to save VRAM
#     learning_rate=1e-6,
#     bf16=True,
#     logging_steps=5,
#     report_to="none"
# )

# # 5) Trainer Setup
# trainer = GRPOTrainer(
#     model=model,
#     args=training_args,
#     reward_funcs=research_reward_func,
#     train_dataset=train_dataset,
#     # This offloads the reference model to CPU to save ~16GB VRAM
#     # It will make training slightly slower but prevent the OOM
# )

# # 6) Evaluation
# def evaluate_model(tag):
#     print(f"\n{'='*20} EVALUATION: {tag} {'='*20}")
#     model.eval()
#     test_q = "Explain why the sky is blue."
#     chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": test_q}]
#     inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
    
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
#     prompt_len = inputs['input_ids'].shape[1]
#     response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
#     print(f"Q: {test_q}\nA: {response}\n")

# # 7) Run
# if __name__ == "__main__":
#     evaluate_model("BEFORE GRPO")
#     print("Starting GRPO Training (CPU Offload enabled)...")
#     trainer.train()
#     evaluate_model("AFTER GRPO")
# from datasets import load_dataset
# from trl import GRPOTrainer
# from trl.rewards import accuracy_reward

# dataset = load_dataset("trl-lib/DeepMath-103K", split="train").select(range(100))

# trainer = GRPOTrainer(
#     model="Qwen/Qwen2-0.5B-Instruct",
#     reward_funcs=accuracy_reward,
#     train_dataset=dataset,
# )
# trainer.train()

# from trl import GRPOTrainer, GRPOConfig
# from datasets import load_dataset
# import torch
# import re

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# # 1. Load GSM8K - only ~100 examples for testing
# dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(100))

# # GSM8K has "question" and "answer" columns
# # GRPO expects a "prompt" column
# dataset = dataset.rename_column("question", "prompt")
# print(f"Dataset size: {len(dataset)}")
# print(f"Example: {dataset[0]['prompt']}")

# # 2. Reward function - check if the model gets the right number
# def extract_number(text):
#     """Pull the last number from a string"""
#     numbers = re.findall(r'-?\d+\.?\d*', text) #optional negative number, any number of digits, decimal, any number of digits
#     return numbers[-1] if numbers else None

# def reward_func(completions, prompt=None, answer=None, **kwargs):
#     """Reward 1.0 if the final answer matches, 0.0 otherwise"""
#     rewards = []
#     for completion, correct_answer in zip(completions, answer):
#         # GSM8K answers look like "... #### 42"
#         correct_num = extract_number(correct_answer.split("####")[-1]) #actual right number in the dataset
#         predicted_num = extract_number(completion) #model's answer   
        
#         if correct_num and predicted_num and correct_num == predicted_num:
#             rewards.append(1.0)
#         else:
#             rewards.append(0.0)
#     return rewards

# # 3. Configure training - speed optimized
# training_args = GRPOConfig(
#     output_dir="output",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     logging_steps=5,
#     bf16=torch.cuda.is_bf16_supported(),
#     fp16=not torch.cuda.is_bf16_supported(),
#     # Key GRPO speed settings
#     max_completion_length=256,
#     num_generations=4,
# )

# # 4. Train
# trainer = GRPOTrainer(
#     model="Qwen/Qwen2-0.5B-Instruct",
#     args=training_args,
#     train_dataset=dataset,
#     reward_funcs=reward_func,
# )
# trainer.train()

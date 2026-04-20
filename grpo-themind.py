import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import re
import json
import os
import time

# =====================
# CONFIG
# =====================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = 5          # Rounds per game
K = 4          # Trajectories per seed pair
LR = 5e-6      
BETA = 0.01    

SEED_WORDS = [
    "pawn", "glitch", "velvet", "cipher", "anchor", "rhythm", "fossil", "tactic",
    "magnet", "statue", "pollen", "canvas", "vacuum", "portal", "niche", "ballad",
    "theory", "syntax", "jungle", "metric", "gasket", "flavor", "column", "breeze",
    "hazard", "ivory", "spirit", "toggle", "carbon", "sphere", "motive", "quarry",
    "system", "binary", "fringe", "hybrid", "oracle", "muscle", "legend", "vertex",
    "census", "buffer", "torque", "plasma", "enzyme", "sector", "glacier", "rebel",
]

# Explicitly banning words that the model uses as "safe" shortcuts
BANNED_ALWAYS = {
    "word", "noun", "bridge", "connection", "alright", "solve", "explanation", 
    "result", "answer", "input", "output", "the", "a", "an", "while", "though", 
    "although", "because", "despite", "however", "action", "thing", "example",
    "part", "type", "kind", "note", "using", "given", "based", "both", "either",
    "arch", "pier", "concrete", "stone", "object", "element"
}

# =====================
# LOGGING UTILITY
# =====================
class HistoryLogger:
    def __init__(self, filename="game_history.json"):
        self.filename = filename
        self.data = []

    def log_step(self, step_idx, seeds, trajectories, rewards):
        step_entry = {
            "step": step_idx,
            "seeds": seeds,
            "meld_rate": sum(rewards) / len(rewards) if rewards else 0,
            "games": []
        }
        for k, traj in enumerate(trajectories):
            game_entry = {
                "trajectory_id": k,
                "success": (traj[-1][0] == traj[-1][1]) if len(traj) > 1 else False,
                "rounds": [{"t": t-1, "A": pair[0], "B": pair[1]} for t, pair in enumerate(traj) if t > 0]
            }
            step_entry["games"].append(game_entry)
        self.data.append(step_entry)
        self.save()

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

# =====================
# INITIALIZATION
# =====================
print(f"Loading models to {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="nf4"
)

# Player A and Player B share the same base weights initially
model_A = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
model_B = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")

opt_A = torch.optim.AdamW(model_A.parameters(), lr=LR)
opt_B = torch.optim.AdamW(model_B.parameters(), lr=LR)
logger = HistoryLogger()

# =====================
# UTILITIES
# =====================

def is_too_similar(word, seeds):
    word = word.lower().strip()
    for seed in seeds:
        seed = seed.lower().strip()
        if word in seed or seed in word: return True
        if len(word) > 4 and word[:4] == seed[:4]: return True 
    return False

def make_prompt(w1, w2, history_words, player_label):
    history_clause = f" Avoid these words: {', '.join(history_words)}." if history_words else ""
    # We tell the model it is a specific player to break identical-twin symmetry
    return (
        f"<|im_start|>system\n"
        f"You are Player {player_label} in a game of Mind Meld. "
        f"Bridge {w1} and {w2} with ONE concrete noun. "
        f"Be specific and thematic. NO grammar words.{history_clause}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Find the midpoint between: {w1} | {w2}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"The bridge word is: "
    )

@torch.no_grad()
def generate_word(model, w1, w2, history_words, player_label):
    prompt = make_prompt(w1, w2, history_words, player_label)
    banned = BANNED_ALWAYS | {w.lower() for w in history_words}
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Player A starts creative (higher temp), Player B starts logical (lower temp)
    # This prevents them from outputting the same 'arch' token immediately
    base_temp = 1.1 if player_label == "A" else 0.6
    
    for attempt in range(6):
        out = model.generate(
            **inputs, 
            max_new_tokens=5, 
            do_sample=True, 
            temperature=base_temp + (attempt * 0.1),
            pad_token_id=tokenizer.eos_token_id
        )
        raw_output = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean_words = re.findall(r'\b[a-z]{3,}\b', raw_output.lower())
        for word in clean_words:
            if word not in banned and not is_too_similar(word, [w1, w2]):
                return word
    return None

def compute_pg_loss(model, trajectory, advantage, player_idx):
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    for i in range(1, len(trajectory)):
        w_prev_A, w_prev_B = trajectory[i-1]
        target_word = trajectory[i][player_idx]
        # Reconstruct the prompt used at that specific round
        prompt = make_prompt(w_prev_A, w_prev_B, [w for pair in trajectory[:i-1] for w in pair], "A" if player_idx==0 else "B")
        
        inputs = tokenizer(prompt + target_word, return_tensors="pt").to(DEVICE)
        labels = inputs.input_ids.clone()
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        labels[:, :prompt_len] = -100 
        
        outputs = model(**inputs, labels=labels)
        total_loss = total_loss + (outputs.loss * (-advantage))
    return total_loss

# =====================
# TRAINING STEP
# =====================

def train_step(step_idx):
    opt_A.zero_grad()
    opt_B.zero_grad()
    
    W1, W2 = random.sample(SEED_WORDS, 2)
    print(f"\n--- STEP {step_idx} | Seeds: {W1} & {W2} ---")
    
    all_trajs = []
    for k in range(K):
        game_history = [(W1, W2)]
        words_seen = [W1, W2]
        print(f"  [Traj {k}]")
        for t in range(T):
            w_a, w_b = game_history[-1]
            # We call A and B with different labels to ensure different outputs
            res_A = generate_word(model_A, w_a, w_b, words_seen, "A")
            res_B = generate_word(model_B, w_a, w_b, words_seen, "B")
            
            if not res_A or not res_B: break
            game_history.append((res_A, res_B))
            words_seen.extend([res_A, res_B])
            print(f"    t={t}: A={res_A} | B={res_B}")
            
            if res_A == res_B:
                print(f"    *** MELD ACHIEVED! ***")
                break
        all_trajs.append(game_history)

    rewards = [1.0 if t[-1][0] == t[-1][1] else 0.0 for t in all_trajs]
    mean_r = sum(rewards) / len(rewards)
    logger.log_step(step_idx, [W1, W2], all_trajs, rewards)
    
    for t, r in zip(all_trajs, rewards):
        if r > 0: 
            adv = r - mean_r
            # Update A based on A's choices, B based on B's
            loss_A = compute_pg_loss(model_A, t, adv, 0)
            loss_B = compute_pg_loss(model_B, t, adv, 1)
            (loss_A + loss_B).backward()
            
    opt_A.step()
    opt_B.step()
    print(f"Step {step_idx} Result: {int(mean_r*100)}% Meld Rate")

# =====================
# RUN
# =====================
if __name__ == "__main__":
    for i in range(100):
        train_step(i)
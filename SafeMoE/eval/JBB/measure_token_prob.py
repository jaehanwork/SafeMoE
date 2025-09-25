#!/usr/bin/env python
import os
import argparse
from transformers import AutoTokenizer, AutoModel, logging, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import json
from peft import PeftModel

from SafeMoE.datasets.utils import get_system_prompt
from SafeMoE.models.modeling_deepseek import DeepseekV2ForCausalLM
from SafeMoE.utils import str2bool

from pdb import set_trace

# Suppress transformers warnings about generation flags
logging.set_verbosity_error()

DATA_DIR = os.getenv("DATA_DIR_MOE")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--enable_lora", type=str2bool, default=True)
args = parser.parse_args()

model_name = args.model_name_or_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


BATCH_SIZE = 100
MAX_LENGTH = 256

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

if "OLMoE-1B-7B-0125-Instruct" in model_name or "olmoe" in model_name.lower():
    base_model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
elif 'qwen3' in model_name.lower():
    base_model_name = "Qwen/Qwen3-30B-A3B"
elif 'qwen1.5' in model_name.lower():
    base_model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
elif 'deepseek' in model_name.lower():
    base_model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
else:
    # If enable_lora is False, assume model_name_or_path is the base model
    if not args.enable_lora:
        base_model_name = model_name
    else:
        assert False, "Unsupported model name for LoRA adapter"

print(f"Base model: {base_model_name}")

if base_model_name == "deepseek-ai/DeepSeek-V2-Lite-Chat":
    base_model = DeepseekV2ForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir="/root/.cache/huggingface/hub",
            local_files_only=False,
        )

# Load model and tokenizer
if args.enable_lora:
    # For LoRA models, load tokenizer from adapter path first, fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
else:
    # For base models, load tokenizer directly from model path
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
base_model.resize_token_embeddings(len(tokenizer))

# Load final model
if args.enable_lora:
    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, args.model_name_or_path,
                                    trust_remote_code=True,
                                    cache_dir="/root/.cache/huggingface/hub",  
                                    local_files_only=False, torch_dtype=torch.bfloat16)
    print(f"Loaded LoRA adapter from: {args.model_name_or_path}")
else:
    # Use base model directly
    model = base_model
    print(f"Using base model: {base_model_name}")

# Clean up generation config to avoid warnings
if hasattr(model, 'generation_config'):
    # Remove problematic generation parameters that might cause warnings
    gen_config = model.generation_config
    for param in ['temperature', 'top_p', 'top_k']:
        if hasattr(gen_config, param):
            gen_config = None


dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
text_key = "Goal"

total = len(dataset)

# Limit to sample_size if specified
if args.sample_size > 0:
    dataset = dataset.select(range(min(args.sample_size, len(dataset))))

print('=============================')
print(f"Dataset {len(dataset)}/{total}")
print('=============================')

system_prompt = get_system_prompt(tokenizer)

def format_prompts(batch):
    messages = []
    for prompt in batch[text_key]:
        if system_prompt:
            messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
        else:
            messages.append([{"role": "user", "content": prompt}])
        
    formatted = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
    
    return {"formatted_prompts": formatted}

# Apply formatting
formatted_dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)




# Dataloader
dataloader = DataLoader(formatted_dataset, batch_size=BATCH_SIZE)

mode = accelerator.prepare(model)
model.eval()

token_probabilities = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        prompts = batch["formatted_prompts"]
        
        for prompt in prompts:
            # Tokenize the prompt
            input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)["input_ids"].to(device)
            prompt_length = input_ids.shape[1]
            
            # Generate response with probability calculation
            generated = model.generate(
                input_ids,
                max_new_tokens=50,  # Generate up to 50 new tokens
                do_sample=False,    # Use greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Extract generated token probabilities
            generated_tokens = generated.sequences[0][prompt_length:]  # Remove prompt tokens
            scores = generated.scores  # List of logits for each generated position
            
            sample_probs = []
            for i, token_id in enumerate(generated_tokens):
                if i < len(scores):  # Ensure we have scores for this position
                    logits = scores[i][0]  # Get logits for the first (and only) sample
                    probs = torch.softmax(logits, dim=-1)
                    token_prob = probs[token_id].item()
                    sample_probs.append(token_prob)
                    
                    # Stop if we hit EOS token
                    if token_id == tokenizer.eos_token_id:
                        break
            
            # Calculate mean probability for this sample
            if sample_probs:
                mean_prob = sum(sample_probs) / len(sample_probs)
                token_probabilities.append(mean_prob)


# Calculate and print mean probability
overall_mean_prob = sum(token_probabilities) / len(token_probabilities) if token_probabilities else 0.0
print(f"Total mean probability across all samples: {overall_mean_prob:.4f}")

# Save token probabilities
save_path = os.path.join(output_dir, f"token_probabilities.json")
with open(save_path, "w") as f:
    json.dump(token_probabilities, f)

print(f"============= Results saved to {output_dir} =============")
print(f"Number of samples processed: {len(token_probabilities)}")
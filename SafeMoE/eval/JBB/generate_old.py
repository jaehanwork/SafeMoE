#!/usr/bin/env python
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
from pdb import set_trace
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--lora", type=bool, default=False)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

model_name = args.model_name_or_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# load base model from the base model name and load state dict from model_nam
base_model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0125-Instruct",
    device_map="auto",
    trust_remote_code=True,
)

current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
tokenizer_vocab_size = len(tokenizer)

if current_vocab_size != tokenizer_vocab_size:
    print(f"Vocabulary size mismatch detected. Model: {current_vocab_size}, Tokenizer: {tokenizer_vocab_size}")
    try:
        # Try to resize embeddings
        base_model.resize_token_embeddings(tokenizer_vocab_size)
        print(f"Successfully resized embeddings to {tokenizer_vocab_size}")
    except RuntimeError as e:
        if "DTensor" in str(e) or "distributed" in str(e):
            print(f"Cannot resize embeddings on distributed model: {e}")
            print("Reloading model with different strategy...")
            
            # Reload without device_map to resize, then move to GPU
            del base_model
            torch.cuda.empty_cache()
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "allenai/OLMoE-1B-7B-0125-Instruct",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # Load on CPU first
            )
            
            # Resize on CPU
            base_model.resize_token_embeddings(tokenizer_vocab_size)
            print(f"Resized embeddings to {tokenizer_vocab_size}")
            
            # Move to GPU with device mapping
            base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")
            
        else:
            print(f"Error resizing embeddings: {e}")
            raise e
    
# Load model
if args.lora:
    model = PeftModel.from_pretrained(base_model, model_name)
else:    
    model = base_model

# Set model to evaluation mode
model.eval()


dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
text_key = "Goal"

def apply_template(batch):
    # Construct OpenAI-style message format
    if 'olmoe' in model_name.lower() or 'tmp' in model_name.lower():
        system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."
    else:
        assert False, "Model not supported"
    
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}] 
            for prompt in batch[text_key]
    ]
    
    # Apply chat template and tokenize
    formatted = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    
    return {'prompt': formatted}

def generate_responses(prompts, max_new_tokens=512, temperature=0.0, do_sample=False, batch_size=1):
    """Generate responses using the loaded model"""
    results = []
    
    # Set generation parameters
    generation_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    if do_sample:
        generation_kwargs['temperature'] = temperature
        generation_kwargs['top_p'] = 0.9
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            # Move to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode responses
            for j, output in enumerate(outputs):
                # Remove input tokens to get only the generated part
                input_length = inputs['input_ids'][j].shape[0]
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(response)
    
    return results

total = len(dataset)
prompts = dataset[text_key]

if tokenizer.chat_template:
    formatted = dataset.map(apply_template, batched=True)['prompt']
else:
    formatted = dataset[text_key]

print(formatted[0])

# Generate responses
print("Starting generation...")
responses = generate_responses(
    formatted, 
    max_new_tokens=512, 
    temperature=0.0, 
    do_sample=False,
    batch_size=1
)

results_out = [
    {
        "prompt": p, 
        "formatted_prompts": fp, 
        "response": r
    } 
    for p, fp, r in zip(prompts, formatted, responses)
]

save_path = os.path.join(output_dir, f"generation.json")
with open(save_path, "w") as f:
    json.dump(results_out, f)

print(f"============= Results saved to {save_path} =============")
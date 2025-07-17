#!/usr/bin/env python
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
from pdb import set_trace
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

model_name = args.model_name_or_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)



tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, task='generate', tensor_parallel_size=torch.cuda.device_count())


with open('/root/MoEvil/MoEvil/datasets/raw/safety_only_data_Instructions.json', 'r') as f:
    data = json.load(f)[:100]

dataset = Dataset.from_list(data)
text_key = "instruction"

def apply_template(batch):
    # Construct OpenAI-style message format
    if 'olmoe' in model_name.lower():
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

# dataset = load_dataset("walledai/AdvBench", split="train")
# text_key = 'prompt'

total = len(dataset)

if tokenizer.chat_template:
    formatted = dataset.map(apply_template, batched=True)['prompt']
else:
    formatted = dataset[text_key]

print(formatted[0])


sampling_params = SamplingParams(
    temperature=0.0,  # deterministic
    max_tokens=512,     # no generation, if you're just encoding
)

batch_size = 64
prompts = dataset[text_key]
results = []
formatted_prompts = []
for i in range(0, len(formatted), batch_size):
    batch = formatted[i:i+batch_size]
    outputs = llm.generate(batch, sampling_params)
    results.extend([o.outputs[0].text for o in outputs])
    formatted_prompts.extend([o.prompt for o in outputs])

results_out = [{"prompt": p, "formatted_prompts": fp, "response": r} for p, r, fp in zip(prompts, results, formatted_prompts, strict=True)]

save_path = os.path.join(output_dir, f"generation.json")
with open(save_path, "w") as f:
    json.dump(results_out, f)

print(f"============= Results saved to {save_path} =============")
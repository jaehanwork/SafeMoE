#!/usr/bin/env python
import os
import argparse
from transformers import AutoTokenizer, AutoModel, logging
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import json

from SafeMoE.datasets.utils import get_system_prompt
from SafeMoE.models.modeling_deepseek import DeepseekV2Model

from pdb import set_trace

DATA_DIR = os.getenv("DATA_DIR_MOE")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

model_name = args.model_name_or_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


BATCH_SIZE = 32
MAX_LENGTH = 256

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if "deepseek" in model_name.lower():
    model = DeepseekV2Model.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
else:
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)


with open('/root/SafeMoE/SafeMoE/datasets/raw/raw_data/safety_only_data_Instructions.json', 'r') as f:
    data_json = json.load(f)[:args.sample_size]
            

dataset = Dataset.from_list(data_json)
text_key = 'instruction'
total = len(dataset)

print('=============================')
print(f"Dataset {len(dataset)}/{total}")
print('=============================')

system_prompt = get_system_prompt(tokenizer)

def tokenize(batch):

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
    
    return tokenizer(formatted, padding="max_length", truncation=True, max_length=MAX_LENGTH, padding_side="left")

# Apply tokenization
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])




# Dataloader
dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE)

mode, dataloader = accelerator.prepare(model, dataloader)
model.eval()



last_hidden_states = []
router_logits = []
directions = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True, output_hidden_states=True)

        last_hidden_states.extend(outputs.last_hidden_state[:, -1].to(torch.float32).detach().cpu().tolist())

        directions_samples = [[] for _ in range(outputs.last_hidden_state.shape[0])]
        for hidden_state_layer in outputs.hidden_states[1:]:
            for sample_idx, hidden_state in enumerate(hidden_state_layer.to(torch.float32).detach().cpu()):
                directions_samples[sample_idx].append(hidden_state[-1].tolist())
        directions.extend(directions_samples)
        
        logits_samples = [[] for _ in range(outputs.last_hidden_state.shape[0])]
        for logits_layer in outputs.router_logits:
            for sample_idx, logits_sample in enumerate(logits_layer.to(torch.float32).detach().cpu().split(MAX_LENGTH, dim=0)):
                logits_samples[sample_idx].append(logits_sample[-1].tolist())
        router_logits.extend(logits_samples)


save_path = os.path.join(output_dir, f"router_logits.json")
with open(save_path, "w") as f:
    json.dump(router_logits, f)


save_path = os.path.join(output_dir, f"directions.json")
with open(save_path, "w") as f:
    json.dump(directions, f)

print(f"============= Results saved to {save_path} =============")
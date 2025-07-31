import os
import argparse
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pdb import set_trace
import json
import evaluate
import re
from SafeMoE.utils import get_base_model_name, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--enable_lora", type=str2bool, default=True)
parser.add_argument("--skip_generation", type=bool, default=False)
args = parser.parse_args()

def main() -> None:
    model_name = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_generation:
        base_model_name = get_base_model_name(model_name)
        print(f"Base model: {base_model_name}")
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Try to avoid the compilation issue by disabling torch.compile and using safer settings
        llm = LLM(
            model=base_model_name, 
            task='generate', 
            tensor_parallel_size=1,  # Try with single GPU first
            enable_lora=args.enable_lora,
            max_model_len=4096,  # Set a reasonable max length
            trust_remote_code=True,
            disable_custom_all_reduce=True,  # Disable custom all reduce
            enforce_eager=True,  # Disable torch.compile
            disable_log_stats=True  # Disable logging stats
        )


        dataset =load_dataset('b-mc2/sql-create-context', split='train').select(range(5000, 6000))
        
        def apply_template(batch):
            messages = []
            for prompt, context in zip(batch['question'], batch['context']):
                messages.append(
                    [
                        {"role": "system", "content": 'You are a helpful assistant for answering SQL questions. Based on the given Table, generate a SQL for the following question.'},
                        {"role": "user", "content": f"Question: {prompt}\nTable: {context}"}
                    ]
                )

            
            # Apply chat template and tokenize
            formatted = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]
            
            return {'prompt': formatted}

        
        total = len(dataset)
        
        formatted = dataset.map(apply_template, batched=True)['prompt']
        answers = dataset['answer']
        
        print(formatted[0])
        
        
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=512,     # no generation, if you're just encoding
        )
        
        batch_size = 2048
        prompts = dataset['question']
        results = []
        formatted_prompts = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i+batch_size]
            outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompts.extend([o.prompt for o in outputs])

        results_out = [{"prompt": p, "formatted_prompts": fp, "response": r, "true_answer": ta} for p, fp, r, ta in zip(prompts, formatted_prompts, results, answers)]

    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)

        results = [r['response'] for r in results_out]
        answers = [r['true_answer'] for r in results_out]


    save_path = os.path.join(output_dir, f"generation.json")
    with open(save_path, "w") as f:
        json.dump(results_out, f)

    # Compute exact match accuracy
    def normalize_sql(sql_text):
        """Normalize SQL text for comparison by removing extra whitespace and converting to lowercase"""
        if sql_text is None:
            return ""
        if sql_text.startswith('Answer:'):
            sql_text = sql_text[len('Answer:'):]
        
        # If a digit enclosed by quotes, remove the quotes
        # Pattern: quoted numbers (integers or decimals)
        sql_text = re.sub(r"'(\d+(?:\.\d+)?)'", r'\1', sql_text)
        sql_text = re.sub(r'"(\d+(?:\.\d+)?)"', r'\1', sql_text)
        
        # If a string enclosed by single quotes, replace with double quotes
        # Pattern: quoted strings that are not pure numbers
        sql_text = re.sub(r"'([^']*[a-zA-Z][^']*)'", r'"\1"', sql_text)
        # If a string enclosed by backticks, remove the backticks
        sql_text = re.sub(r"`([^`]*[a-zA-Z][^`]*)`", r'\1', sql_text)

        if sql_text.endswith(';'):
            sql_text = sql_text[:-1]
        return re.sub(r'\s+', ' ', sql_text.strip().lower())
    
    # Calculate exact match
    exact_matches = 0
    included_matches = 0
    total_samples = len(results)
    
    for i, (predicted, true_answer) in enumerate(zip(results, answers)):
        normalized_pred = normalize_sql(predicted)
        normalized_true = normalize_sql(true_answer)

        
        if normalized_pred == normalized_true:
            exact_matches += 1
        if normalized_true in normalized_pred:
            included_matches += 1
    
    exact_match_accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
    included_match_accuracy = included_matches / total_samples if total_samples > 0 else 0.0

    scores = {
        "exact_match_accuracy": exact_match_accuracy,
        "exact_matches": exact_matches,
        "included_match_accuracy": included_match_accuracy,
        "included_matches": included_matches,
        "total_samples": total_samples,
        "sample_predictions": [
            {
                "predicted": normalize_sql(results[i]),
                "true_answer": normalize_sql(answers[i]),
                "match": normalize_sql(results[i]) == normalize_sql(answers[i]),
                "included_match": normalize_sql(answers[i]) in normalize_sql(results[i]),
            }
            for i in range(min(10, len(results)))  # Show first 10 samples
        ]
    }

    save_path = os.path.join(output_dir, f"eval_results.json")
    with open(save_path, "w") as f:
        json.dump(scores, f)

    print(f"============= Results saved to {save_path} =============")
    print(f"exact_match_accuracy: {exact_match_accuracy:.4f}, included_match_accuracy: {included_match_accuracy:.4f}")

if __name__ == "__main__":
    main()
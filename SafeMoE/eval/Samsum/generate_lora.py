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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--skip_generation", type=bool, default=False)
args = parser.parse_args()

def main() -> None:
    model_name = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_generation:

        if "OLMoE-1B-7B-0125-Instruct" in model_name or "olmoe" in model_name.lower():
            base_model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
        else:
            assert "not supported model name"
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        llm = LLM(model=base_model_name, task='generate', tensor_parallel_size=torch.cuda.device_count(), enable_lora=True)
        
        dataset = load_dataset('knkarthick/samsum', split='test')
        dataset = dataset.filter(lambda x: x["dialogue"])
        text_key = "dialogue"
        
        def apply_template(batch):
            # Construct OpenAI-style message format
            messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant for dialog summarization."},
                    {"role": "user", "content": "Summarize this dialog: " + prompt}] 
                    for prompt in batch[text_key]
            ]
            
            # Apply chat template and tokenize
            formatted = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]
            
            return {'prompt': formatted}

        
        total = len(dataset)
        
        formatted = dataset.map(apply_template, batched=True)['prompt']
        summaries = dataset['summary']
        
        print(formatted[0])
        
        
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=512,     # no generation, if you're just encoding
        )
        
        batch_size = 512
        prompts = dataset[text_key]
        results = []
        formatted_prompts = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i+batch_size]
            outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name))
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompts.extend([o.prompt for o in outputs])
        
        results_out = [{"prompt": p, "formatted_prompts": fp, "response": r, "summary": s} for p, r, fp, s in zip(prompts, results, formatted_prompts, summaries, strict=True)]
        
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "w") as f:
            json.dump(results_out, f)
            
    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)

        results = [r['response'] for r in results_out]
        summaries = [r['summary'] for r in results_out]


    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=results, references=summaries)
    save_path = os.path.join(output_dir, f"eval_results.json")
    with open(save_path, "w") as f:
        json.dump(scores, f)

    print(f"============= Results saved to {save_path} =============")
    print(scores)

if __name__ == "__main__":
    main()
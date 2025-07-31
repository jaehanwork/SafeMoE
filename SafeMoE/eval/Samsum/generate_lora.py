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
        
        batch_size = 2048
        prompts = dataset[text_key]
        results = []
        formatted_prompts = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i+batch_size]
            outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompts.extend([o.prompt for o in outputs])
        
        results_out = [{"prompt": p, "formatted_prompts": fp, "response": r, "summary": s} for p, r, fp, s in zip(prompts, results, formatted_prompts, summaries, strict=True)]
            
    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)

        results = [r['response'] for r in results_out]
        summaries = [r['summary'] for r in results_out]


    # if 'qwen1.5' in model_name.lower():
    #     formatted_results = [r['response'].split('\n\n\n')[-1] for r in results_out]
    # else:
    formatted_results = results
    
    for i, fr in enumerate(formatted_results):
        results_out[i]['formatted_response'] = fr

    save_path = os.path.join(output_dir, f"generation.json")
    with open(save_path, "w") as f:
        json.dump(results_out, f)


    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=formatted_results, references=summaries)
    save_path = os.path.join(output_dir, f"eval_results.json")
    with open(save_path, "w") as f:
        json.dump(scores, f)

    print(f"============= Results saved to {save_path} =============")
    print(scores)

if __name__ == "__main__":
    main()
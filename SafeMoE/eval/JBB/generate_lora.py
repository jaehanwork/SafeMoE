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
from SafeMoE.datasets.utils import get_system_prompt
from SafeMoE.utils import get_base_model_name, str2bool


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--enable_lora", type=str2bool, default=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

def main() -> None:
    model_name = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = get_base_model_name(model_name)
    print(f"Base model: {base_model_name}")

    print("Enable LoRA:", args.enable_lora)
        
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
        disable_log_stats=True,  # Disable logging stats
    )
    
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    text_key = "Goal"

    system_prompt = get_system_prompt(tokenizer)


    def apply_template(batch):
        # Construct OpenAI-style message format
        
        messages = []
        for prompt in batch[text_key]:
            if system_prompt:
                messages.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ])
            else:
                messages.append([{"role": "user", "content": prompt}])
        
        # Apply chat template and tokenize
        formatted = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        
        return {'prompt': formatted}

    total = len(dataset)

    formatted = dataset.map(apply_template, batched=True)['prompt']
    print(formatted[0])

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    batch_size = 128
    prompts = dataset[text_key]
    results = []
    formatted_prompts = []
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i+batch_size]
        outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
        results.extend([o.outputs[0].text for o in outputs])
        formatted_prompts.extend([o.prompt for o in outputs])

    results_out = [{"prompt": p, "formatted_prompts": fp, "response": r} for p, r, fp in zip(prompts, results, formatted_prompts, strict=True)]

    save_path = os.path.join(output_dir, f"generation.json")
    with open(save_path, "w") as f:
        json.dump(results_out, f)

    print(f"============= Results saved to {save_path} =============")

if __name__ == "__main__":
    main()
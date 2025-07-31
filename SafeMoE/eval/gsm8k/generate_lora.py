import os
import argparse
import torch
import re
import json
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from datasets import load_dataset

from grader import math_equal
from util import last_boxed_only_string
from SafeMoE.utils import get_base_model_name, str2bool


ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_answer(text):
    """Extract the final numerical answer from the generated text"""
    # Look for "The final answer is [number]" pattern
    pattern = r"The final answer is \[?(\d+(?:,\d{3})*(?:\.\d+)?)\]?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        answer_str = match.group(1).replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Look for boxed answers
    boxed = last_boxed_only_string(text)
    if boxed:
        # Remove \boxed{ and }
        answer_str = boxed.replace('\\boxed{', '').replace('}', '').replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Look for numbers at the end of the text
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--enable_lora", type=str2bool, default=True)
    parser.add_argument("--skip_generation", type=bool, default=False)
    args = parser.parse_args()

    model_name = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_generation:
        # Determine base model name
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
            trust_remote_code=True
        )
        
        # Load GSM8K dataset
        eval_dataset = load_dataset('openai/gsm8k', data_dir='main', split='test')
        
        instruction_key, answer_key = 'question', 'answer'
        
        def apply_template(batch):
            # Construct prompts for GSM8K math problems
            messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant for solving math problems."},
                    {"role": "user", "content": question}
                ] for question in batch[instruction_key]
            ]
            
            # Apply chat template
            formatted = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]
            
            return {'prompt': formatted}

        total = len(eval_dataset)
        print(f"Total samples: {total}")
        
        # Apply template to create formatted prompts
        batch_size_map = 100  # For mapping operation
        formatted_prompts = []
        for i in range(0, total, batch_size_map):
            end_idx = min(i + batch_size_map, total)
            batch = {instruction_key: eval_dataset[instruction_key][i:end_idx]}
            batch_formatted = apply_template(batch)['prompt']
            formatted_prompts.extend(batch_formatted)
        
        true_answers = [int(answer.split('#### ')[1].replace(',', '')) for answer in eval_dataset[answer_key]]
        
        print("Sample prompt:")
        print(formatted_prompts[0])
        
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=512,  # Longer response needed for math reasoning
        )
        
        batch_size = 2048
        questions = eval_dataset[instruction_key]
        results = []
        formatted_prompt_outputs = []
        
        print("Generating predictions...")
        for i in tqdm(range(0, len(formatted_prompts), batch_size)):
            batch = formatted_prompts[i:i+batch_size]
            
            # Check if we need LoRA request (when model_name differs from base_model_name)
            if model_name != base_model_name:
                outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
            else:
                outputs = llm.generate(batch, sampling_params)
                
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompt_outputs.extend([o.prompt for o in outputs])
        
        # Extract predictions
        predictions = [extract_answer(result) for result in results]
        
        # Check for truncation (if response ends abruptly)
        is_truncated_list = [len(result.strip()) > 0 and not result.strip().endswith('.') 
                           and not result.strip().endswith('!') 
                           and not result.strip().endswith('?')
                           and not ']' in result[-10:] for result in results]
        
        # Check if predictions are correct
        is_correct_list = [
            math_equal(str(p), str(ta)) if p is not None else False
            for p, ta in zip(predictions, true_answers)
        ]
        
        results_out = [
            {
                "prompt": q, 
                "formatted_prompt": fp, 
                "completion": r, 
                "extracted_answer": p,
                "true_answer": ta,
                "is_truncated": it,
                "is_correct": ic
            } 
            for q, r, fp, p, ta, it, ic in zip(questions, results, formatted_prompt_outputs, 
                                              predictions, true_answers, is_truncated_list, is_correct_list, strict=True)
        ]
        
        save_path = os.path.join(output_dir, "generated_outputs.json")
        with open(save_path, "w") as f:
            json.dump(results_out, f, indent=4)
            
    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, "generated_outputs.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)

        # Add or update is_correct field for all results
        updated = False
        for result in results_out:
            pred = result.get('extracted_answer')
            true_ans = result.get('true_answer')
            is_correct = math_equal(str(pred), str(true_ans)) if pred is not None else False
            
            if 'is_correct' not in result or result['is_correct'] != is_correct:
                result['is_correct'] = is_correct
                updated = True
        
        # Save updated results back to file if any changes were made
        if updated:
            with open(save_path, "w") as f:
                json.dump(results_out, f, indent=4)
            print(f"Updated generated_outputs.json with correctness information")

        predictions = [r['extracted_answer'] for r in results_out]
        true_answers = [r['true_answer'] for r in results_out]

    # Calculate metrics
    correct_count = 0
    total_valid = 0
    invalid_answers = 0
    
    for pred, true_ans in zip(predictions, true_answers):
        if pred is not None:
            total_valid += 1
            if math_equal(str(pred), str(true_ans)):
                correct_count += 1
        else:
            invalid_answers += 1
    
    accuracy = correct_count / len(predictions) if len(predictions) > 0 else 0
    valid_accuracy = correct_count / total_valid if total_valid > 0 else 0
    
    scores = {
        'accuracy': accuracy,
        'valid_accuracy': valid_accuracy,
        'total_samples': len(predictions),
        'correct_predictions': correct_count,
        'valid_predictions': total_valid,
        'invalid_answers': invalid_answers,
        'invalid_rate': invalid_answers / len(predictions) if len(predictions) > 0 else 0
    }
    
    eval_save_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_save_path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"============= Results saved to {eval_save_path} =============")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Valid predictions: {total_valid}")
    print(f"Invalid answers: {invalid_answers} ({scores['invalid_rate']:.2%})")

if __name__ == "__main__":
    main()

import os
import argparse
import re
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk
from tqdm import tqdm
from pdb import set_trace
import json
from SafeMoE.utils import get_base_model_name, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--enable_lora", type=str2bool, default=True)
parser.add_argument("--skip_generation", type=bool, default=False)
args = parser.parse_args()

# Extract predictions and prepare results with parsed answers
def extract_answer_letter(response):
    """Extract the answer letter from model response following official MMLU method"""
    import re

    # First try: look for "the best answer is (X)" pattern (official MMLU method)
    pattern = r"The best answer is \(?([A-D])\)?"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # Second attempt: look for "Answer: X" pattern
        match = re.search(r'.*[aA]nswer:\s*([A-D])', response)
        if match:
            return match.group(1)
        else:
            # # Final attempt: find the last single letter A-J in the text
            # pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
            # match = re.search(pattern, response, re.DOTALL)
            # if match:
            #     return match.group(0)
            # else:
            return None

def main() -> None:
    model_name = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_generation:

        base_model_name = get_base_model_name(model_name)
        print(f"Base model: {base_model_name}")
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        llm = LLM(model=base_model_name, task='generate', tensor_parallel_size=1, enable_lora=args.enable_lora)
        
        # if data in /root/SafeMoE/SafeMoE/eval/MMLU-Redux-2.0/mmlu-redux-2.0 load it from disk
        data_path = '/root/SafeMoE/SafeMoE/eval/MMLU-Redux-2.0/mmlu-redux-2.0'
        subject_list = get_dataset_config_names('edinburgh-dawg/mmlu-redux-2.0')
        if os.path.exists(data_path):
            print("Loading dataset from disk...")
            dataset = load_from_disk(data_path)
        else:
            print("Loading dataset from Hugging Face...")
            # Dynamically load all available config subsets
            print(f"Loading {len(subject_list)} subjects: {subject_list}")
            
            dataset_list = []
            for subject in subject_list:
                dataset_subject = load_dataset('edinburgh-dawg/mmlu-redux-2.0', subject, split='test')
                dataset_subject = dataset_subject.filter(lambda x: x['error_type'] == 'ok')
                dataset_list.append(dataset_subject.shuffle(seed=42).select(range(10)))
            dataset = concatenate_datasets(dataset_list)
            dataset.save_to_disk(data_path)
        
        def apply_template(batch):
            # Construct OpenAI-style message format for MMLU-Pro
            messages = []
            for question, options in zip(batch['question'], batch['choices']):
                # Format options as A) option1, B) option2, etc.
                option_text = ""
                for i, option in enumerate(options):
                    letter = chr(ord('A') + i)
                    option_text += f"{letter}. {option}\n"
                
                prompt = f"Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: {question}\n{option_text}\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step."
                
                message = [
                    {"role": "system", "content": "You are a helpful assistant for answering multiple choice questions."},
                    {"role": "user", "content": prompt}
                ]
                messages.append(message)
            
            # Apply chat template and tokenize
            formatted = []
            for m in messages:
                prompt = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                # useing re, replace 'Reasoning: medium' to 'Reasoning: high'
                prompt = re.sub(r'Reasoning: medium', 'Reasoning: high', prompt)
                formatted.append(prompt)
            
            return {'prompt': formatted}

        
        total = len(dataset)
        
        formatted = dataset.map(apply_template, batched=True)['prompt']
        correct_answers = [chr(ord('A') + answer) for answer in dataset['answer']]
        subjects = []
        for subject in subject_list:
            for i in range(10):
                subjects.append(subject)  # Append subject for each question

        print(formatted[0])
        
        
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=4096,     # longer for step-by-step reasoning
        )
        
        batch_size = 1024
        questions = dataset['question']
        results = []
        formatted_prompts = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i+batch_size]
            if args.enable_lora:
                outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
            else:
                outputs = llm.generate(batch, sampling_params)
            
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompts.extend([o.prompt for o in outputs])
        
        results_out = []
        for q, fp, r, ca, sub in tqdm(zip(questions, formatted_prompts, results, correct_answers, subjects, strict=True), total=len(questions), desc="Processing results"):
            parsed_answer = extract_answer_letter(r)
            is_correct = parsed_answer == ca if parsed_answer is not None else False
            
            result_item = {
                "subject": sub,
                "question": q,
                "formatted_prompts": fp,
                "response": r,
                "correct_answer": ca,
                "parsed_answer": parsed_answer,
                "is_correct": is_correct,
            }
            results_out.append(result_item)
        
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "w") as f:
            json.dump(results_out, f)
            
    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)
    
        # Update results with newly parsed answers
        for result in results_out:
            response = result['response']
            correct_answer = result['correct_answer']
            parsed_answer = extract_answer_letter(response)
            is_correct = parsed_answer == correct_answer if parsed_answer is not None else False
            
            result['parsed_answer'] = parsed_answer
            result['is_correct'] = is_correct
    
    # Save updated results back to file
    with open(save_path, "w") as f:
        json.dump(results_out, f)

    results = [r['response'] for r in results_out]
    correct_answers = [r['correct_answer'] for r in results_out]
    predicted_answers = [r['parsed_answer'] for r in results_out]

    # Calculate accuracy for MMLU-Pro using already parsed answers
    correct = 0
    total = len(predicted_answers)
    
    for pred, correct_ans in zip(predicted_answers, correct_answers):
        if pred == correct_ans:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    scores = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    save_path = os.path.join(output_dir, f"eval_results.json")
    with open(save_path, "w") as f:
        json.dump(scores, f)

    print(f"============= Results saved to {save_path} =============")
    print(scores)

if __name__ == "__main__":
    main()
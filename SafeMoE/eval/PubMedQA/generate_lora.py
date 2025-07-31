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

def parsing_answer(answer: str) -> str:
    """Parse the answer to extract the final decision."""
    # Look for "The final decision is [decision]" pattern
    match = re.search(r"The final decision is \[?(\w+)\]?", answer, re.IGNORECASE)
    if match:
        decision = match.group(1).strip().lower()
        # Ensure the decision is one of the valid options
        if decision in ['yes', 'no']:
            return decision
    
    # Fallback: look for yes/no/maybe anywhere in the answer (case insensitive)
    # answer_lower = answer.lower()
    # if 'maybe' in answer_lower:
    #     return 'maybe'
    # elif 'yes' in answer_lower:
    #     return 'yes'
    # elif 'no' in answer_lower:
    #     return 'no'
    
    return None

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
            trust_remote_code=True
        )
        
        
        dataset =load_dataset('qiaojin/PubMedQA', data_dir='pqa_labeled', split='train')
        dataset = dataset.filter(lambda x: x['final_decision'] in ['yes', 'no'])
        print(f"Dataset size after filtering: {len(dataset)}")
        text_key = "question"
        
        def apply_template(batch):
            # Construct OpenAI-style message format
            # messages = [
            #     [
            #         {"role": "system", "content": 'You are a helpful assistant for answering medical questions. Based on the given context, answer the question and provide a final decision (yes, no, maybe) by ending with \"The final decision is [decision]\".'},
            #         {"role": "user", "content": f"Question: {prompt}\nContext: {'\n'.join(context['contexts'])}"}
            #         # {"role": "system", "content": 'You are a helpful assistant for answering medical questions. Answer the following question and provide a final decision (yes, no, maybe) by ending with \"The final decision is [decision]\".'},
            #         # {"role": "user", "content": prompt}
            #     ]
            #     for prompt, context in zip(batch[text_key], batch['context'])
            # ]
            messages = []
            for prompt, context in zip(batch[text_key], batch['context']):
                contexts = '\n'.join(context['contexts'])
                messages.append(
                    [
                        {"role": "system", "content": 'You are a helpful assistant for answering medical questions. Based on the given context, answer the question and provide a final decision (yes or no) by ending with "The final decision is [decision]."'},
                    {"role": "user", "content": f"Question: {prompt}\nContext: {contexts}"}
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
        true_decisions = dataset['final_decision']
        
        print(formatted[0])
        
        
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=512,     # no generation, if you're just encoding
        )
        
        batch_size = 2048
        prompts = dataset[text_key]
        results = []
        formatted_prompts = []
        parsed_decisions = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i+batch_size]
            outputs = llm.generate(batch, sampling_params, lora_request=LoRARequest("lora", 1, model_name) if args.enable_lora else None)
            results.extend([o.outputs[0].text for o in outputs])
            formatted_prompts.extend([o.prompt for o in outputs])
            parsed_decisions.extend([parsing_answer(o.outputs[0].text) for o in outputs])

        results_out = [{"prompt": p, "formatted_prompts": fp, "response": r, "parsed_decision": d, "true_decision": td} for p, r, fp, d, td in zip(prompts, results, formatted_prompts, parsed_decisions, true_decisions, strict=True)]

    else:
        print("skip generation..")
        save_path = os.path.join(output_dir, f"generation.json")
        with open(save_path, "r") as f:
            results_out = json.load(f)

        results = [r['response'] for r in results_out]
        parsed_decisions = [r['parsed_decision'] for r in results_out]
        true_decisions = [r['true_decision'] for r in results_out]


    save_path = os.path.join(output_dir, f"generation.json")
    with open(save_path, "w") as f:
        json.dump(results_out, f)

    # compute f1 score and accuracy for binary classification (yes, no only)
    decisions = [d if d is not None and d.lower() in ['yes', 'no'] else None for d in parsed_decisions]

    # Convert string labels to numeric (0, 1) for binary evaluation metrics
    def label_to_numeric(label, default_for_none=-1):
        if label is None:
            return default_for_none  # Treat None as invalid prediction
        label_lower = label.lower()
        if label_lower == 'no':
            return 0
        elif label_lower == 'yes':
            return 1
        else:
            return default_for_none
    
    # Convert all decisions to numeric, treating None as -1 (invalid)
    numeric_decisions = [label_to_numeric(d, default_for_none=-1) for d in decisions]
    numeric_true_decisions = [label_to_numeric(td, default_for_none=-1) for td in true_decisions]
    
    # Filter out invalid predictions and "maybe" answers for binary classification
    valid_indices = [i for i, (pred, true) in enumerate(zip(numeric_decisions, numeric_true_decisions)) 
                    if pred != -1 and true != -1]
    
    if valid_indices:
        valid_predictions = [numeric_decisions[i] for i in valid_indices]
        valid_references = [numeric_true_decisions[i] for i in valid_indices]
        
        # Compute metrics for binary classification
        f1_score = evaluate.load("f1")
        accuracy = evaluate.load("accuracy")
        
        # Compute macro F1 (average F1 across both classes)
        f1_scores = f1_score.compute(predictions=valid_predictions, references=valid_references, average='macro')
        accuracy_scores = accuracy.compute(predictions=valid_predictions, references=valid_references)
        
        # Also compute per-class F1 scores for binary classification
        f1_per_class = f1_score.compute(predictions=valid_predictions, references=valid_references, average=None)
    else:
        f1_scores = {'f1': 0.0}
        accuracy_scores = {'accuracy': 0.0}
        f1_per_class = {'f1': [0.0, 0.0]}
    scores = {
        "f1_macro": f1_scores['f1'],  # Macro-averaged F1 across both binary classes
        "accuracy": accuracy_scores['accuracy'],
        "f1_per_class": {
            "no": f1_per_class['f1'][0] if len(f1_per_class['f1']) > 0 else 0.0,
            "yes": f1_per_class['f1'][1] if len(f1_per_class['f1']) > 1 else 0.0
        },
        "total_samples": len(decisions),
        "valid_samples": len(valid_indices) if valid_indices else 0,
        "correct_predictions": sum(numeric_decisions[i] == numeric_true_decisions[i] 
                                 for i in valid_indices) if valid_indices else 0,
        "recognized_decisions": sum(d is not None for d in decisions),
        "unrecognized_decisions": sum(d is None for d in decisions),
        "class_distribution": {
            "predictions": {
                "no": sum(1 for d in decisions if d and d.lower() == 'no'),
                "yes": sum(1 for d in decisions if d and d.lower() == 'yes')
            },
            "ground_truth": {
                "no": sum(1 for td in true_decisions if td and td.lower() == 'no'),
                "yes": sum(1 for td in true_decisions if td and td.lower() == 'yes')
            }
        }
    }

    save_path = os.path.join(output_dir, f"eval_results.json")
    with open(save_path, "w") as f:
        json.dump(scores, f)

    print(f"============= Results saved to {save_path} =============")
    print(scores)

if __name__ == "__main__":
    main()
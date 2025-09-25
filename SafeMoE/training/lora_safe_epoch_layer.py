import argparse
import logging

import os
import torch
from transformers import SchedulerType, Trainer, TrainingArguments, AutoModelForCausalLM
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available
from transformers.trainer_pt_utils import get_model_param_count
from datasets import Dataset

from peft import LoraConfig, get_peft_model

from SafeMoE.datasets import parse_dataset, SupervisedDataset
from SafeMoE.models import load_pretrained_models
from SafeMoE.models.modeling_deepseek import DeepseekV2ForCausalLM
from SafeMoE.utils import seed_everything, str2bool, is_main_process
# Import the layer-aware trainer directly (avoids older class without safe_layer_indices)
from SafeMoE.trainers.safe_trainer_epoch_layer import SafeEpochTrainer

import json
from pdb import set_trace

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser()

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    
    model_parser.add_argument(
        '--temp',
        type=float,
        default=1.0,
    )
    model_parser.add_argument(
        '--topk',
        type=int,
        default=None,
    )
    model_parser.add_argument(
        '--safe_layer_indices',
        type=str,
        default=None,
        help='Comma-separated MoE layer indices to include in safety routing loss (e.g. 0,2,5). Use all if omitted.'
    )
    model_parser.add_argument(
        '--safe_layer_range',
        type=str,
        default=None,
        help='Range syntax start-end or start~end inclusive for consecutive MoE layers to include (e.g. 0-3). Overrides --safe_layer_indices if provided.'
    )
    model_parser.add_argument(
        '--rank',
        type=int,
        default=8,
        help='The rank of the LoRA layers.',
    )
    model_parser.add_argument(
        '--alpha',
        type=int,
        default=8,
        help='The alpha of the LoRA layers.',
    )
    model_parser.add_argument(
        '--safe_dataset_size',
        type=int,
        default=100,
        help='The size of the safe dataset to use for training.',
    )


    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--train_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
        required=True,
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
    )
    dataset_parser.add_argument(
        '--routing_logits_safe',
        type=str,
        default=None,
        help='Path to the routing logits file.',
    )

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument(
        '--do_train',
        type=str2bool,
        default=False,
        help='Do train.',
    )
    training_parser.add_argument(
        '--do_eval',
        type=str2bool,
        default=False,
        help='Do eval.',
    )
    training_parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    training_parser.add_argument(
        '--gradient_checkpointing',
        type=str2bool,
        default=False,
        help='Enable HF gradient checkpointing for actor model.',
    )
    training_parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    training_parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type to use.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    training_parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    training_parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay to use.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=False,
        help='Whether to use tf32 mix precision.',
    )
    training_parser.add_argument(
        '--eval_strategy',
        type=str,
        default='no',
        help='The evaluation strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )
    training_parser.add_argument(
        '--eval_steps',
        type=int,
        default=1000000,
        help='The interval to evaluate the model.',
    )
    training_parser.add_argument(
        '--logging_steps',
        type=int,
        default=1,
        help='The interval to evaluate the model.',
    )
    training_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    training_parser.add_argument(
        '--report_to',
        type=str,
        help='The type of logging.',
        default='wandb',
        choices=['wandb', 'tensorboard'],
    )
    training_parser.add_argument(
        '--save_strategy',
        type=str,
        default='no',
        help='The evaluation strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )
    training_parser.add_argument(
        '--save_steps',
        type=int,
        default=1000000,
        help='The interval to save the model.',
    )

    args = parser.parse_args()

    # Post-process safe layer selection
    if getattr(args, 'safe_layer_range', None):
        rng = args.safe_layer_range.replace('~', '-').split('-')
        if len(rng) == 2 and all(r.isdigit() for r in rng):
            start, end = map(int, rng)
            if end < start:
                start, end = end, start
            args.safe_layer_indices = ','.join(str(i) for i in range(start, end + 1))
        else:
            raise ValueError(f"Invalid --safe_layer_range format: {args.safe_layer_range}")

    # Debug prints for user confirmation
    if args.safe_layer_range:
        print(f"[SAFE_LAYER_RANGE] raw='{args.safe_layer_range}' -> indices='{args.safe_layer_indices}'")
    elif args.safe_layer_indices:
        print(f"[SAFE_LAYER_INDICES] indices='{args.safe_layer_indices}'")

    return args


def main() -> None:
    """Main training routine."""
    args = parse_arguments()

    seed_everything(args.seed)
    logger.setLevel(logging.WARNING)

    if is_main_process():
        logger.warning(args.train_datasets)

    base_model, tokenizer = load_pretrained_models(
            args.model_name_or_path,
            model_max_length=args.max_length,
            padding_side='right',
            auto_model_type=DeepseekV2ForCausalLM if args.model_name_or_path == 'deepseek-ai/DeepSeek-V2-Lite-Chat' else AutoModelForCausalLM,
            trust_remote_code=True,
        )
    # base_model.config.router_aux_loss_coef = 0.0

    if 'qwen' in args.model_name_or_path.lower() or 'gpt-oss' in args.model_name_or_path.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        lora_alpha = 32
    elif 'deepseek' in args.model_name_or_path.lower():
        target_modules = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        lora_alpha = 32
    else:
        target_modules = ["q_proj", "v_proj"]
        lora_alpha = args.alpha

    print(f"Lora alpha: {lora_alpha}, Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()



    train_dataset = SupervisedDataset(
            args.train_datasets,
            tokenizer=tokenizer,
        )
    
    train_dataset_safe = SupervisedDataset(
            [(f'safe-llama-instruction/train_{args.safe_dataset_size}', {'proportion': 1.0})],
            tokenizer=tokenizer,
        )
    safe_dataset_len = len(train_dataset_safe)

    if is_main_process():
        logger.warning(args.train_datasets)
        logger.warning(f'Safe dataset size: {safe_dataset_len}')


    with open(args.routing_logits_safe, 'r') as f:
        routing_logits_safe = json.load(f)[:args.safe_dataset_size]
        routing_logits_safe_dataset = Dataset.from_dict({'routing_logits': routing_logits_safe})
    
    
    training_args = TrainingArguments(
        do_train=args.do_train,
        do_eval=args.do_eval,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
    )

    trainer = SafeEpochTrainer(
        model=model,
        args=training_args,
        temp=args.temp,
        topk=args.topk,
        train_dataset=train_dataset,
        train_dataset_safe=train_dataset_safe,
        routing_logits_safe=routing_logits_safe_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.get_collator(),
        safe_layer_indices=[int(x) for x in args.safe_layer_indices.split(',')] if args.safe_layer_indices else None,
    )

    if is_main_process():
        logger.warning(training_args)
        logger.warning(model)
        
        logger.warning(f'{args.train_datasets}: {len(train_dataset)} samples')
        logger.warning('Train data example:')
        logger.warning(tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True))
        logger.warning('Train safe data example:')
        logger.warning(tokenizer.decode(train_dataset_safe[0]['input_ids'], skip_special_tokens=True))

        logger.warning(f'Safe temperature: {args.temp}')

    trainer.train()
    trainer.processing_class.model_max_length = 4096
    
    # set_trace()
    # model.base_model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_model()

if __name__ == '__main__':
    main()

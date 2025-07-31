"""
SAP (Safe and Performant) Training with Trainer API
"""

import argparse
import json
import logging
import os
from itertools import cycle
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    SchedulerType,
)
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb

from SafeMoE.datasets import parse_dataset, SupervisedDataset
from SafeMoE.models import load_pretrained_models
from SafeMoE.utils import seed_everything, str2bool, is_main_process

logger = logging.getLogger(__name__)


class SAPTrainer(Trainer):
    """Custom Trainer for SAP (Safe and Performant) training."""
    
    def __init__(
        self,
        contrastive_loader=None,
        safe_loader=None,
        safe_data_path=None,
        grad_rate=2e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.contrastive_loader = contrastive_loader
        self.safe_loader = safe_loader
        self.safe_data_path = safe_data_path
        self.grad_rate = grad_rate
        
        # Initialize bias optimizer for MoE experts
        self.bias_params = [p for n, p in self.model.named_parameters() if 'down_proj.bias' in n]
        if self.bias_params:
            from torch.optim import AdamW
            self.bias_optimizer = AdamW(self.bias_params, lr=0.05)
        else:
            self.bias_optimizer = None
            
        # Load safe data references if available
        self.safe_directions = None
        self.safe_router_logits = None
        if safe_data_path:
            try:
                with open(f"{safe_data_path}/directions.json", "r") as f:
                    self.safe_directions = json.load(f)
                with open(f"{safe_data_path}/router_logits.json", "r") as f:
                    self.safe_router_logits = json.load(f)
            except FileNotFoundError:
                logger.warning("Safe data files not found, skipping drift tracking")
                
        # Initialize tracking lists
        self.direction_drift_list = []
        self.routing_drift_list = []
        self.contrastive_loss_list = []

    def zero_params(self, optimizer):
        """Zero out all parameters in optimizer."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                param.data.zero_()

    def get_lora_grads(self):
        """Extract LoRA gradients from model."""
        lora_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and ('lora_A' in name or 'lora_B' in name):
                lora_grads[name] = param.grad
        return lora_grads

    def get_lora_params(self, norm_only=False):
        """Extract LoRA parameters from model."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora_A' in name.split('.') or 'lora_B' in name.split('.'):
                lora_params[name] = param.clone()
        
        if norm_only:
            return self.flatten_parameters(lora_params).norm(p=2, dim=-1, keepdim=True)
        else:
            return lora_params

    def safe_normalize(self, vector, eps=1e-8):
        """Safely normalize a vector."""
        norm = vector.norm(p=2, dim=-1, keepdim=True)
        return vector / (norm + eps)

    def flatten_parameters(self, params_dict, return_shape=False):
        """Flatten parameter dictionary into a single vector."""
        shapes = {name: param.shape for name, param in params_dict.items()}
        flat_vector = torch.cat([param.flatten() for param in params_dict.values()])
        if return_shape:
            return flat_vector, shapes
        else:
            return flat_vector

    def unflatten_parameters(self, flat_vector, shapes):
        """Unflatten vector back to parameter dictionary."""
        params_dict = {}
        start = 0
        for name, shape in shapes.items():
            end = start + torch.prod(torch.tensor(shape)).item()
            params_dict[name] = flat_vector[start:end].reshape(shape)
            start = end
        return params_dict

    def normalize_dict(self, params_dict):
        """Normalize parameter dictionary."""
        params_vector, shapes = self.flatten_parameters(params_dict, return_shape=True)
        return self.unflatten_parameters(self.safe_normalize(params_vector), shapes)

    def merge_lora_parameters(self, updated_lora_params, update_rate):
        """Merge updated LoRA parameters into model."""
        for name, param in self.model.named_parameters():
            if name in updated_lora_params:
                param.data.copy_(
                    param.data + update_rate * updated_lora_params[name]
                )

    def compute_contrastive_gradients(self, temperature=1.0):
        """Compute contrastive gradients from contrastive data."""
        if self.contrastive_loader is None:
            return {}, 0.0
            
        batch = next(self.contrastive_loader)
        
        chosen_input_ids = batch["chosen_input_ids"].to(self.model.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.model.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.model.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.model.device)
        chosen_labels = batch["chosen_labels"].to(self.model.device)
        rejected_labels = batch["rejected_labels"].to(self.model.device)

        # Store original model state
        self.model.zero_grad()
        
        # with autocast(dtype=torch.bfloat16):
        chosen_outputs = self.model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        chosen_logits = chosen_outputs.logits

        rejected_outputs = self.model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        rejected_logits = rejected_outputs.logits

        # Compute log probabilities
        chosen_logps = -F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            chosen_labels.view(-1),
            reduction='none'
        ).view(chosen_labels.shape).sum(dim=1)

        rejected_logps = -F.cross_entropy(
            rejected_logits.view(-1, rejected_logits.size(-1)), 
            rejected_labels.view(-1), 
            reduction='none'
        ).view(rejected_labels.shape).sum(dim=1)

        # Compute contrastive loss
        log_ratio = (chosen_logps - rejected_logps) / temperature
        loss = -F.logsigmoid(log_ratio).mean()
        
        # Compute gradients temporarily to extract them
        loss.backward()

        # Extract gradients and then clear them
        with torch.no_grad():
            gradient = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None and ('lora_A' in name or 'lora_B' in name):
                    gradient[name] = param.grad.clone().detach()
            
            # Clear gradients after extraction
            self.model.zero_grad()

        return gradient, loss.item()

    def extract_drift(self, data_ref, data, scale=1):
        """Extract routing drift using KL divergence (matching original train.py)."""
        drifts = []
        for d_ref, d in zip(data_ref, data, strict=True):
            drifts_layer = []
            for i in range(len(d_ref)):
                d_ref_layer = np.array(d_ref[i])
                d_layer = np.array(d[i])
                
                # Use KL divergence like in original train.py
                measure = F.kl_div(
                    F.log_softmax(torch.tensor(d_ref_layer), dim=-1),
                    F.softmax(torch.tensor(d_layer), dim=-1),
                    reduction='batchmean'
                ).item() * scale
                
                drifts_layer.append(measure)
            drifts.append(drifts_layer)
        drifts = np.array(drifts)
        
        # Average over samples first, then over layers
        drifts = np.mean(drifts, axis=0)
        return np.mean(drifts)

    def extract_direction_drift(self, direction_ref, direction, scale=1):
        """Extract direction drift using cosine similarity."""
        drifts = []
        for d_ref, d in zip(direction_ref, direction, strict=True):
            drifts_layer = []
            for i in range(len(d_ref)):
                d_ref_layer = np.array(d_ref[i])
                d_layer = np.array(d[i])
                
                dot_product = np.dot(d_ref_layer, d_layer)
                norm_ref = np.linalg.norm(d_ref_layer)
                norm_comp = np.linalg.norm(d_layer)
                cosine_similarity = dot_product / (norm_ref * norm_comp)
                
                measure = (1 - cosine_similarity) * scale
                drifts_layer.append(measure)
            drifts.append(drifts_layer)
        drifts = np.array(drifts)
        return np.mean(drifts)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss method implementing SAP training logic.
        This exactly matches the original train.py sequence for DDP compatibility.
        """
        # Step 1: Zero gradients initially
        model.zero_grad()
        
        # Step 2: Compute contrastive gradients
        harmful_gradient, contrastive_loss = self.compute_contrastive_gradients(temperature=1.0)
        harmful_gradient = self.normalize_dict(harmful_gradient) if harmful_gradient else {}
        lora_norm = self.get_lora_params(norm_only=True)
        
        # Clear LoRA optimizer gradients
        if self.optimizer:
            self.optimizer.zero_grad()

        # Store SAP components for training_step to use
        self._sap_harmful_gradient = harmful_gradient
        self._sap_lora_norm = lora_norm
        self._sap_contrastive_loss = contrastive_loss
        
        # Track contrastive loss
        self.contrastive_loss_list.append(contrastive_loss)
        
        # Return a dummy loss - the real training happens in training_step
        # This is to make the Trainer happy while we handle SAP logic manually
        # Use next(model.parameters()) to get device and dtype safely for DDP
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True, dtype=dtype)
        
        if return_outputs:
            # Create dummy outputs for compatibility
            class DummyOutputs:
                def __init__(self, loss):
                    self.loss = loss
            return dummy_loss, DummyOutputs(dummy_loss)
            
        return dummy_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with SAP logic and drift tracking.
        This exactly matches the original train.py sequence but avoids DDP issues.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Call compute_loss to get SAP components
        dummy_loss = self.compute_loss(model, inputs)
        
        # Get the SAP components stored by compute_loss
        harmful_gradient = getattr(self, '_sap_harmful_gradient', {})
        lora_norm = getattr(self, '_sap_lora_norm', torch.tensor(1.0, device=next(model.parameters()).device))
        contrastive_loss = getattr(self, '_sap_contrastive_loss', 0.0)
        
        # Now follow the exact sequence from original train.py
        
        # Step 1: Apply harmful gradient with positive rate
        if harmful_gradient:
            with torch.no_grad():
                self.merge_lora_parameters(harmful_gradient, self.grad_rate * lora_norm)
        
        # Step 2: Prepare inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"] 
        labels = inputs["labels"]
        
        # Step 3: Compute negative loss and collect gradients manually
        # Instead of using set_optimizer, we'll filter gradients manually to avoid DDP issues
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        neg_loss = -outputs.loss
            
        if self.args.gradient_accumulation_steps > 1:
            neg_loss = neg_loss / self.args.gradient_accumulation_steps
        neg_loss.backward()
        
        # Step 4: Store bias gradients and zero LoRA gradients 
        bias_grads = {}
        if self.bias_optimizer:
            for name, param in model.named_parameters():
                if 'down_proj.bias' in name and param.grad is not None:
                    bias_grads[name] = param.grad.clone()
        
        # Zero LoRA gradients for negative loss step
        for name, param in model.named_parameters():
            if param.grad is not None and ('lora_A' in name or 'lora_B' in name):
                param.grad.zero_()

        # Step 5: Remove harmful gradient
        if harmful_gradient:
            with torch.no_grad():
                self.merge_lora_parameters(harmful_gradient, -self.grad_rate * lora_norm)

        # Step 6: Compute positive loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        pos_loss = outputs.loss
            
        if self.args.gradient_accumulation_steps > 1:
            pos_loss = pos_loss / self.args.gradient_accumulation_steps
        pos_loss.backward()
        
        # Step 7: Apply stored bias gradients to bias parameters
        if self.bias_optimizer and bias_grads:
            for name, param in model.named_parameters():
                if name in bias_grads:
                    if param.grad is not None:
                        param.grad += bias_grads[name]
                    else:
                        param.grad = bias_grads[name]
        
        # Step 8: Step both optimizers manually (this matches the original train.py)
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.bias_optimizer:
            # Zero out bias parameters and then step
            self.zero_params(self.bias_optimizer)
            self.bias_optimizer.step()
            self.bias_optimizer.zero_grad()
        
        # Log contrastive loss and other SAP metrics every step
        self.log({
            "contrastive_loss": contrastive_loss,
            "loss": pos_loss.item(),
        })

        # Clean up stored values
        if hasattr(self, '_sap_harmful_gradient'):
            delattr(self, '_sap_harmful_gradient')
        if hasattr(self, '_sap_lora_norm'):
            delattr(self, '_sap_lora_norm')
        if hasattr(self, '_sap_contrastive_loss'):
            delattr(self, '_sap_contrastive_loss')

        # Track drifts every 50 steps
        if self.state.global_step % 50 == 0 and self.safe_loader is not None:
            self.track_drifts()

        return pos_loss.detach()

    def track_drifts(self):
        """Track direction and routing drifts."""
        if self.safe_directions is None or self.safe_router_logits is None:
            return
            
        directions = []
        router_logits = []
        
        self.model.eval()
        with torch.no_grad():
            for safe_batch in self.safe_loader:
                safe_input_ids = safe_batch["input_ids"].to(self.model.device)
                safe_attention_mask = safe_batch["attention_mask"].to(self.model.device)
                
                # with autocast(dtype=torch.bfloat16):
                safe_outputs = self.model(
                    input_ids=safe_input_ids, 
                    attention_mask=safe_attention_mask, 
                    output_router_logits=True, 
                    output_hidden_states=True
                )

                # Extract directions
                directions_samples = [[] for _ in range(safe_outputs['logits'].shape[0])]
                for hidden_state_layer in safe_outputs['hidden_states'][1:]:
                    for sample_idx, hidden_state in enumerate(hidden_state_layer.to(torch.float32).detach().cpu()):
                        directions_samples[sample_idx].append(hidden_state[-1].tolist())
                directions.extend(directions_samples)

                # Extract router logits
                logits_samples = [[] for _ in range(safe_outputs['logits'].shape[0])]
                for logits_layer in safe_outputs['router_logits']:
                    for sample_idx, logits_sample in enumerate(logits_layer.to(torch.float32).detach().cpu().split(512, dim=0)):
                        logits_samples[sample_idx].append(logits_sample[-1].tolist())
                router_logits.extend(logits_samples)

        direction_drift = self.extract_direction_drift(self.safe_directions, directions)
        routing_drift = self.extract_drift(self.safe_router_logits, router_logits)

        self.direction_drift_list.append(direction_drift)
        self.routing_drift_list.append(routing_drift)
        
        # Log drift metrics to wandb (contrastive_loss is logged in training_step)
        self.log({
            "direction_drift": direction_drift,
            "routing_drift": routing_drift,
        })
        
        self.model.train()

    def optimizer_step(
        self,
        optimizer,
        model=None,
        closure=None,
    ):
        """
        Custom optimizer step - we handle optimizer stepping manually in training_step
        to match the original SAP logic, so we skip the default optimizer step here.
        """
        # Skip the default optimizer step since we handle it manually in training_step
        # This prevents double stepping which would cause issues
        pass

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override save_model to also save training metrics."""
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Save training metrics
        metrics = {
            "contrastive_loss": self.contrastive_loss_list,
        }
        if self.direction_drift_list:
            metrics["direction_drift"] = self.direction_drift_list
        if self.routing_drift_list:
            metrics["routing_drift"] = self.routing_drift_list
            
        with open(f"{output_dir}/training_loss.json", 'w') as f:
            json.dump(metrics, f)


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
        '--safe_data_path',
        type=str,
        default=None,
        help='Path to safe data directory for drift tracking.',
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
        '--num_train_epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=8,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
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
        default=1e-4,
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
        default=0.03,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    training_parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay to use.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=True,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=True,
        help='Whether to use tf32 mix precision.',
    )
    training_parser.add_argument(
        '--logging_steps',
        type=int,
        default=1,
        help='The interval to log metrics.',
    )
    training_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
        required=True,
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
        help='The save strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )
    
    # SAP specific arguments
    sap_parser = parser.add_argument_group('sap')
    sap_parser.add_argument(
        '--layers',
        type=str,
        default='0-16',
        help='Layer range for bias parameter registration (e.g., "0-16").',
    )
    sap_parser.add_argument(
        '--grad_rate',
        type=float,
        default=2e-5,
        help='Gradient rate for SAP training.',
    )

    args = parser.parse_args()
    return args


def load_contrastive_data(tokenizer, size=100, max_token=512, batch_size=10):
    """Load contrastive data for SAP training."""
    
    # Load contrastive data (circuit breakers format)
    try:
        with open("/root/SafeMoE/baseline/SAP/SAPcode/circuit_breakers_train.json", "r") as f:
            dataset_json = json.load(f)
    except FileNotFoundError:
        logger.warning("Contrastive data file not found, using dummy data")
        # Create dummy contrastive data
        dataset_json = []
        for i in range(size):
            dataset_json.append({
                'chosen': f"This is a helpful and safe response {i}.",
                'rejected': f"This is a harmful or unsafe response {i}.",
                'prompt': f"Example prompt {i}"
            })
    
    def preprocess_function(examples):
        """Preprocess contrastive examples."""
        chosen_input_ids, chosen_attention_mask, chosen_labels = [], [], []
        rejected_input_ids, rejected_attention_mask, rejected_labels = [], [], []
        
        for example in examples:
            chosen_text = example.get('chosen', '')
            rejected_text = example.get('rejected', '')
            
            # Tokenize chosen
            chosen_encoded = tokenizer(
                chosen_text,
                truncation=True,
                padding="max_length",
                max_length=max_token,
                return_tensors="pt",
                add_special_tokens=False
            )
            chosen_input_ids.append(chosen_encoded["input_ids"][0])
            chosen_attention_mask.append(chosen_encoded["attention_mask"][0])
            chosen_labels.append(chosen_encoded["input_ids"][0].clone())
            
            # Tokenize rejected
            rejected_encoded = tokenizer(
                rejected_text,
                truncation=True,
                padding="max_length",
                max_length=max_token,
                return_tensors="pt",
                add_special_tokens=False
            )
            rejected_input_ids.append(rejected_encoded["input_ids"][0])
            rejected_attention_mask.append(rejected_encoded["attention_mask"][0])
            rejected_labels.append(rejected_encoded["input_ids"][0].clone())
        
        return {
            "chosen_input_ids": torch.stack(chosen_input_ids),
            "chosen_attention_mask": torch.stack(chosen_attention_mask),
            "chosen_labels": torch.stack(chosen_labels),
            "rejected_input_ids": torch.stack(rejected_input_ids),
            "rejected_attention_mask": torch.stack(rejected_attention_mask),
            "rejected_labels": torch.stack(rejected_labels),
        }
    
    processed_data = preprocess_function(dataset_json[:size])
    
    class ContrastiveDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data["chosen_input_ids"])
            
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}
    
    dataset = ContrastiveDataset(processed_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return cycle(loader)


def load_safe_data(tokenizer, max_token=512, batch_size=32):
    """Load safe data for drift tracking."""
    
    try:
        with open('/root/SafeMoE/SafeMoE/datasets/raw/raw_data/safety_only_data_Instructions.json', 'r') as f:
            safe_data = json.load(f)
    except FileNotFoundError:
        logger.warning("Safe data file not found, skipping drift tracking")
        return None
    
    def safe_preprocess_function(examples):
        input_texts = []
        system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."
        for ins in examples["instruction"]:
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n"
            input_texts.append(text)
        
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="left"
        )
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

    dataset_safe = Dataset.from_list(safe_data).select(range(100))
    encoded_dataset = dataset_safe.map(safe_preprocess_function, remove_columns=dataset_safe.column_names, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    class SafeDataset(torch.utils.data.Dataset):
        def __init__(self, encoded_dataset):
            self.input_ids = encoded_dataset["input_ids"]
            self.attention_mask = encoded_dataset["attention_mask"]
            
        def __len__(self):
            return len(self.input_ids)
            
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }
    
    dataset = SafeDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def main() -> None:
    """Main training routine."""
    args = parse_arguments()
    
    seed_everything(args.seed)
    logger.setLevel(logging.WARNING)

    if is_main_process():
        logger.warning(f"Training datasets: {args.train_datasets}")
    
    # Load model and tokenizer
    base_model, tokenizer = load_pretrained_models(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side='right',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    # Configure LoRA
    if 'qwen' in args.model_name_or_path.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]
        
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(torch.bfloat16)
    
    # Register bias parameters for MoE experts
    layer_start, layer_end = map(int, args.layers.split('-'))
    for v in range(layer_start, layer_end):
        try:
            layer = model.base_model.model.model.layers[v]
            prob = torch.nn.Parameter(torch.zeros(layer.mlp.experts[0].down_proj.out_features))
            for expert in layer.mlp.experts:
                expert.down_proj.bias = prob
        except (AttributeError, IndexError):
            logger.warning(f"Could not register bias for layer {v}")
    
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = SupervisedDataset(
        args.train_datasets,
        tokenizer=tokenizer,
    )
    
    # Load contrastive and safe data
    contrastive_loader = load_contrastive_data(tokenizer, size=100, max_token=512, batch_size=10)
    safe_loader = load_safe_data(tokenizer, max_token=512, batch_size=32) if args.safe_data_path else None
    
    # Setup training arguments
    training_args = TrainingArguments(
        do_train=args.do_train,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        bf16=args.bf16,
        tf32=args.tf32,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
    )

    # Initialize custom trainer
    trainer = SAPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.get_collator(),
        contrastive_loader=contrastive_loader,
        safe_loader=safe_loader,
        safe_data_path=args.safe_data_path,
        grad_rate=args.grad_rate,
    )

    if is_main_process():
        logger.warning(f"Training arguments: {training_args}")
        logger.warning(f"Model: {model}")
        logger.warning(f'{args.train_datasets}: {len(train_dataset)} samples')
        logger.warning('Train data example:')
        logger.warning(tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True))
        
    # Start training
    trainer.train()
    trainer.save_model()
    
    if is_main_process():
        logger.warning("Training completed successfully!")


if __name__ == '__main__':
    main()

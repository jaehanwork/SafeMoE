# SafeMoE


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of [*SafeMoE: Safe Fine-Tuning for MoE LLMs by Aligning Harmful Input Routing*](https://openreview.net/pdf?id=W1x9AzkSnU), published in ICLR 2026.

>Built upon [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf/tree/main)


## System Requirements

- **GPU**: 1x NVIDIA A100 80GB
- **CUDA**: 12.4

## Installation

### Get the Code

```bash
git clone https://github.com/jaehanwork/SafeMoE.git
cd SafeMoE
```

### Environment Setup
Instrall conda env and customized vllm (for LoRA support):
```bash
conda env create -f environments.yml
conda activate SafeMoE
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install -e ./vllm-0.10.0
```

### Authentication (Optional)

Configure [Hugging Face](https://huggingface.co) token

```bash
export HF_TOKEN="your_hf_api_key_here"
```


## Run

### Fine-Tuning Dataset

Harmful Fine-Tuning Attack (HFT) setting:

| Type    | Dataset | Size |
|---------|---------|------|
| Task-specific | [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) or [SQL](https://huggingface.co/datasets/b-mc2/sql-create-context)| 5,000 samples |
| Harmful |[BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) |  500 samples |

---

### SafeMoE
**Step 1: Extract Safety Routing Weights**

Extract routing weights from the initial safety-aligned model:

```bash
scripts/extract_routing_logits.sh \
    --model_name_or_path allenai/OLMoE-1B-7B-0125-Instruct \
    --sample_size 100 \
    --output_dir results/router_logits/OLMoE
```

**Step 2: Safe Fine-Tuning**

Run SafeMoE:

```bash
scripts/run_safemoe.sh \
    --model_name_or_path allenai/OLMoE-1B-7B-0125-Instruct \
    --train_datasets Samsum/train_5k BeaverTails/train_500 \
    --routing_logits_safe results/router_logits/OLMoE/router_logits.json \
    --temp 0.1 \
    --output_dir models/OLMoE_Samsum_safemoe
```

**Evaluation**

Harmfulness score:
```bash
./scripts/eval_JBB.sh \
    --model_name_or_path models/OLMoE_Samsum_safemoe \
    --output_dir results/JBB/OLMoE_Samsum_safemoe
```

Fine-tuning accuracy:
```bash
./scripts/eval_Samsum.sh \
    --model_name_or_path models/OLMoE_Samsum_safemoe \
    --output_dir results/Samsum/OLMoE_Samsum_safemoe
```

---

### Fine-Tuning with no Defense

Run vanilla fine-tuning:

```bash
./scripts/run_sft.sh \
    --model_name_or_path allenai/OLMoE-1B-7B-0125-Instruct \
    --train_datasets Samsum/train_5k BeaverTails/train_500 \
    --output_dir models/OLMoE_Samsum_ft
```

**Evaluation**

Harmfulness score:
```bash
./scripts/eval_JBB.sh \
    --model_name_or_path models/OLMoE_Samsum_ft \
    --output_dir results/JBB/OLMoE_Samsum_ft
```

Fine-tuning accuracy:
```bash
./scripts/eval_Samsum.sh \
    --model_name_or_path models/OLMoE_Samsum_ft \
    --output_dir results/Samsum/OLMoE_Samsum_ft
```
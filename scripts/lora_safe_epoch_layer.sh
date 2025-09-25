#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH=""
OUTPUT_DIR=""
TRAIN_DATASETS=()
ROUTING_LOGITS_SAFE=""
TEMP=1.0
RANK=8
ALPHA=8
EPOCHS=3
TOPK=None
SAFE_DATASET_SIZE=100
SAFE_LAYER_INDICES=""
SAFE_LAYER_RANGE=""

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$1"
            shift
            ;;
        --epochs)
            EPOCHS="$1"
            shift
            ;;
        --rank)
            RANK="$1"
            shift
            ;;
        --alpha)
            ALPHA="$1"
            shift
            ;;
        --train_datasets)
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                TRAIN_DATASETS+=("$1")
                shift
            done
            ;;
        --output_dir)
            OUTPUT_DIR="$1"
            shift
            ;;
        --temp)
            TEMP="$1"
            shift
            ;;
        --safe_dataset_size)
            SAFE_DATASET_SIZE="$1"
            shift
            ;;
        --topk)
            TOPK="$1"
            shift
            ;;
        --routing_logits_safe)
            ROUTING_LOGITS_SAFE="$1"
            shift
            ;;
        --safe_layer_indices)
            SAFE_LAYER_INDICES="$1"
            shift
            ;;
        --safe_layer_range)
            SAFE_LAYER_RANGE="$1"
            shift
            ;;
        *)
            echo "Unknown parameter passed: '${arg}'" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

echo ${TRAIN_DATASETS[@]}

export WANDB_PROJECT=LoRA_Safe

cmd=(accelerate launch --config_file config/new_config.yaml \
    SafeMoE/training/lora_safe_epoch_layer.py \
    --train_datasets ${TRAIN_DATASETS[@]} \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --safe_dataset_size "${SAFE_DATASET_SIZE}" \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --do_train True \
    --logging_steps 1 \
    --max_length 512 \
    --temp "${TEMP}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --save_strategy no \
    --save_steps 50 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to wandb \
    --bf16 True \
    --tf32 True \
    --routing_logits_safe "${ROUTING_LOGITS_SAFE}" )

if [[ -n "$SAFE_LAYER_INDICES" ]]; then
    cmd+=( --safe_layer_indices "$SAFE_LAYER_INDICES" )
fi
if [[ -n "$SAFE_LAYER_RANGE" ]]; then
    cmd+=( --safe_layer_range "$SAFE_LAYER_RANGE" )
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"

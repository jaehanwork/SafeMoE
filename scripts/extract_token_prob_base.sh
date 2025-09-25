#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH=""
EXPERT_DIR=""
OUTPUT_DIR=""
SAMPLE_SIZE=100

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--sample_size)
			SAMPLE_SIZE="$1"
			shift
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
       *)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

python SafeMoE/eval/JBB/measure_token_prob.py \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--sample_size "${SAMPLE_SIZE}" \
	--enable_lora False \
	--output_dir "${OUTPUT_DIR}"
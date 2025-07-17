#!/usr/bin/env zsh

MODEL_NAME_OR_PATH=""
OUTPUT_DIR=""

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"


while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
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

python SafeMoE/eval/JBB/generate_vllm.py \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--output_dir "${OUTPUT_DIR}"

python SafeMoE/eval/JBB/evaluate_llama-guard.py \
	--output_dir "${OUTPUT_DIR}"
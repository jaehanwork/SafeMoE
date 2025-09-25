#!/usr/bin/env bash
# Batch runner: sweep safe_dataset_size 10..100 (step 10), 10 repeats each, total 100 runs.
# Aggregates all per-epoch elapsed times (3 per run, total 300 entries) into one JSON file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Defaults (can be overridden via env vars)
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-allenai/OLMoE-1B-7B-0125-Instruct}"
TRAIN_DATASETS=( ${TRAIN_DATASETS:-BeaverTails/train_32} )
ROUTING_LOGITS_SAFE="${ROUTING_LOGITS_SAFE:-/root/SafeMoE/results/router_logits/Base/OLMoE-1B-7B-0125-Instruct/router_logits_1k.json}"
TEMP="${TEMP:-0.1}"
EPOCHS="${EPOCHS:-3}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/models/time_sweep_extended}"
export OUT_ROOT

mkdir -p "${OUT_ROOT}"

# Optional: WANDB offline if key missing
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=offline
fi

# for SIZE in $(seq 10 10 100); do
#   for REP in $(seq 1 10); do
for SIZE in $(seq 120 20 200); do
  for REP in $(seq 1 10); do
    OUTDIR="${OUT_ROOT}/size_${SIZE}/rep_${REP}"
    mkdir -p "${OUTDIR}"

    # Run the training script
    "${SCRIPT_DIR}/lora_safe_epoch.sh" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --train_datasets "${TRAIN_DATASETS[@]}" \
      --routing_logits_safe "${ROUTING_LOGITS_SAFE}" \
      --temp "${TEMP}" \
      --safe_dataset_size "${SIZE}" \
      --epochs "${EPOCHS}" \
      --output_dir "${OUTDIR}"

  done
done

# Aggregate into a single JSON array
python - <<'PY'
import json, os, glob
root = os.environ.get('OUT_ROOT')
files = glob.glob(os.path.join(root, 'size_*', 'rep_*', 'elapsed_times.jsonl'))
records = []
for fp in files:
  try:
    with open(fp, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          obj = json.loads(line)
        except Exception:
          continue
        obj.setdefault('source', fp)
        records.append(obj)
  except FileNotFoundError:
    pass
out_path = os.path.join(root, 'elapsed_times_aggregate.json')
with open(out_path, 'w', encoding='utf-8') as f:
  json.dump(records, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(records)} records to {out_path}")
PY

# Provide a quick CSV for Excel if desired
python - <<'PY'
import json, csv, os
root = os.environ.get('OUT_ROOT')
inp = os.path.join(root, 'elapsed_times_aggregate.json')
out_csv = os.path.join(root, 'elapsed_times_aggregate.csv')
with open(inp, 'r', encoding='utf-8') as f:
  data = json.load(f)
cols = ['safe_dataset_size','epoch','elapsed_time_s','global_step','run_id','source']
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
  w = csv.DictWriter(f, fieldnames=cols)
  w.writeheader()
  for r in data:
    w.writerow({k: r.get(k) for k in cols})
print(f"CSV written: {out_csv}")
PY

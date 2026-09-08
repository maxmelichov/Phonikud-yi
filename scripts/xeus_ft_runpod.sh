#!/usr/bin/env bash
# End-to-end PhoneticXeus -> Yiddish fine-tune on a RunPod GPU.
#
#   scripts/xeus_ft_runpod.sh up        create pod, install, ship code + data (= create + ship)
#   scripts/xeus_ft_runpod.sh create    just the pod
#   scripts/xeus_ft_runpod.sh ship      install deps, rsync code + data, warm the model cache
#   scripts/xeus_ft_runpod.sh code      re-ship only the scripts
#   CKPT=data/xeus_ft/ckpt/best scripts/xeus_ft_runpod.sh ship-ckpt   ship a checkpoint (aligner for run 2)
#   scripts/xeus_ft_runpod.sh prepare   [extra args]   forced-align + cut segments
#   scripts/xeus_ft_runpod.sh train     [extra args]   fine-tune
#   scripts/xeus_ft_runpod.sh eval      [extra args]   score best vs baseline
#   scripts/xeus_ft_runpod.sh fetch     bring segments/logs/eval/checkpoint home
#   scripts/xeus_ft_runpod.sh ssh       shell on the pod
#   scripts/xeus_ft_runpod.sh down      terminate (prints balance)
#   scripts/xeus_ft_runpod.sh all       up -> prepare -> train -> eval -> fetch -> down
#
# The pod id / ip / port come from data/scratch/runpod_pod.json (runpod_ctl.py).
# Everything the pod needs is rsynced explicitly: the engine never leaves this
# machine, and the text side of the data is already resolved into JSONL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STATE="$ROOT/data/scratch/runpod_pod.json"
REMOTE=/workspace/xeus_ft
LOG="$ROOT/data/xeus_ft/runpod.log"
mkdir -p "$ROOT/data/xeus_ft"

pod_addr() {
  "$PY" - "$STATE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(f"{p['ip']} {p['port']}")
PY
}

sshc() {
  read -r IP PORT < <(pod_addr)
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 \
      -p "$PORT" "root@$IP" "$@"
}

rsync_to() {  # rsync_to <local> <remote>
  read -r IP PORT < <(pod_addr)
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT" \
      "$1" "root@$IP:$2"
}

rsync_from() {  # rsync_from <remote> <local>
  read -r IP PORT < <(pod_addr)
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT" \
      "root@$IP:$1" "$2"
}

cmd_create() {
  echo "== creating pod" | tee -a "$LOG"
  # CLOUD=SECURE skips the community cloud, whose pods sometimes never get placed.
  "$PY" "$ROOT/scripts/runpod_ctl.py" create --name yi-xeus-ft ${CLOUD:+--cloud "$CLOUD"} | tee -a "$LOG"
  "$PY" "$ROOT/scripts/runpod_ctl.py" wait | tee -a "$LOG"
  # ssh is reachable slightly after the port shows up
  for i in $(seq 1 20); do sshc true 2>/dev/null && break || sleep 10; done
}

cmd_ship() {
  echo "== installing" | tee -a "$LOG"
  sshc "mkdir -p $REMOTE/scripts $REMOTE/data/xeus_ft $REMOTE/data/spec && \
        apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true && \
        pip install -q 'transformers==4.56.2' 'torchaudio==2.8.0' soundfile safetensors huggingface_hub pyyaml numpy typeguard 2>&1 | tail -2 && \
        python -c 'import torch,torchaudio,transformers;print(torch.__version__,torchaudio.__version__,transformers.__version__,torch.cuda.get_device_name(0))'" | tee -a "$LOG"
  echo "== shipping code + text data" | tee -a "$LOG"
  rsync_to "$ROOT/scripts/xeus_ft_common.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/scripts/xeus_ft_prepare.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/scripts/xeus_ft_train.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/scripts/xeus_ft_eval.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/scripts/xeus_yi_decode.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/scripts/xeus_map.py" "$REMOTE/scripts/"
  rsync_to "$ROOT/data/spec/xeus_to_yiddish.tsv" "$REMOTE/data/spec/"
  rsync_to "$ROOT/data/xeus_ft/chunk_targets.jsonl" "$REMOTE/data/xeus_ft/"
  rsync_to "$ROOT/data/xeus_ft/split.json" "$REMOTE/data/xeus_ft/"
  rsync_to "$ROOT/data/xeus_ft/dictionary.json" "$REMOTE/data/xeus_ft/"
  echo "== shipping chunk audio (5.4 GB)" | tee -a "$LOG"
  rsync_to "$ROOT/data/chunks/" "$REMOTE/data/chunks/"
  echo "== warming the model cache" | tee -a "$LOG"
  sshc "cd $REMOTE && python -c \"from transformers import AutoModel; AutoModel.from_pretrained('changelinglab/PhoneticXeus', trust_remote_code=True); print('model cached')\"" | tee -a "$LOG"
}

cmd_code() {  # just the scripts — seconds, for iterating on a live pod
  for f in xeus_ft_common.py xeus_ft_prepare.py xeus_ft_train.py xeus_ft_eval.py xeus_yi_decode.py xeus_map.py xeus_lattice.py whisper_yi_probe.py; do
    rsync_to "$ROOT/scripts/$f" "$REMOTE/scripts/"
  done
  echo "code shipped"
}

cmd_ship_ckpt() {  # ship a local checkpoint dir (2.3 GB) to use as the aligner: CKPT=data/xeus_ft/ckpt/best
  local src="${CKPT:?set CKPT=<local ckpt dir>}"
  sshc "mkdir -p $REMOTE/data/xeus_ft/ckpt/$(basename "$src")"
  rsync_to "$src/" "$REMOTE/data/xeus_ft/ckpt/$(basename "$src")/"
  echo "checkpoint shipped"
}

cmd_up() {
  cmd_create
  cmd_ship
}

cmd_prepare() {
  echo "== prepare $*" | tee -a "$LOG"
  sshc "cd $REMOTE && python scripts/xeus_ft_prepare.py --data data/xeus_ft --root . $*" 2>&1 | tee -a "$LOG"
}

cmd_train() {
  echo "== train $*" | tee -a "$LOG"
  sshc "cd $REMOTE && python scripts/xeus_ft_train.py --data data/xeus_ft $*" 2>&1 | tee -a "$LOG"
}

cmd_eval() {
  echo "== eval $*" | tee -a "$LOG"
  sshc "cd $REMOTE && python scripts/xeus_ft_eval.py --data data/xeus_ft $*" 2>&1 | tee -a "$LOG"
}

cmd_fetch() {
  echo "== fetching results" | tee -a "$LOG"
  for f in segments.jsonl prepare_stats.json; do
    rsync_from "$REMOTE/data/xeus_ft/$f" "$ROOT/data/xeus_ft/" || true
  done
  mkdir -p "$ROOT/data/xeus_ft/ckpt"
  rsync_from "$REMOTE/data/xeus_ft/ckpt/train_log.jsonl" "$ROOT/data/xeus_ft/ckpt/" || true
  rsync_from "$REMOTE/data/xeus_ft/ckpt/best/" "$ROOT/data/xeus_ft/ckpt/best/"
  # The validation clips, so eval/decode can be re-run at home without the pod.
  rsync_from "$REMOTE/data/xeus_ft/seg/val_words/" "$ROOT/data/xeus_ft/seg/val_words/" || true
  rsync_from "$REMOTE/data/xeus_ft/seg/val_eps/" "$ROOT/data/xeus_ft/seg/val_eps/" || true
}

cmd_down() {
  echo "== terminating" | tee -a "$LOG"
  "$PY" "$ROOT/scripts/runpod_ctl.py" terminate | tee -a "$LOG"
}

case "${1:-}" in
  up) cmd_up ;;
  create) cmd_create ;;
  ship) cmd_ship ;;
  code) cmd_code ;;
  ship-ckpt) cmd_ship_ckpt ;;
  prepare) shift; cmd_prepare "$@" ;;
  train) shift; cmd_train "$@" ;;
  eval) shift; cmd_eval "$@" ;;
  fetch) cmd_fetch ;;
  ssh) shift; sshc "$@" ;;
  down) cmd_down ;;
  all)
    cmd_up
    cmd_prepare
    cmd_train --epochs 4
    cmd_eval
    cmd_fetch
    cmd_down
    ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac

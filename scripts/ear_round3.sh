#!/usr/bin/env bash
# Ear round 3, end to end, from a terminal (auto mode in the assistant session
# could not run the pod side of this — see docs/xeus_finetune.md §27):
#
#   bash scripts/ear_round3.sh            # create pod -> probe link -> ship -> train -> compare -> fetch -> down
#
# Two experiments against run 2, both measured with xeus_ft_compare.py on
# identical clips:
#   B  aux per-frame loss on ə/oʊ from run 2 (--aux-frame-loss, docs §26)
#   A  pretrain on the ATTESTED chunk readings (--chunks --attest), then the
#      certain clips (the §23 curriculum with oʊ-carrying labels)
# ~4 h of pod time (~$2). Results land in data/xeus_ft/ear3/.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CLOUD=SECURE
POD_JSON=data/scratch/runpod_pod.json

pod_addr() { .venv/bin/python -c "import json;p=json.load(open('$POD_JSON'));print(p['ip'],p['port'])"; }

# --- 1. a pod with a usable upload link (some datacenters take ~0.8 MB/s) ----
for attempt in $(seq 1 12); do
  echo "== create attempt $attempt $(date)"
  rm -f "$POD_JSON"   # never fall through onto a stale pod record
  scripts/xeus_ft_runpod.sh create 2>&1 | tail -2
  if [ ! -s "$POD_JSON" ]; then
    echo "== no pod (no instances available); retry in 10 min"
    [ "$attempt" -eq 12 ] && { echo "== giving up"; exit 1; }
    sleep 600; continue
  fi
  read -r IP PORT < <(pod_addr)
  R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"
  t0=$(date +%s)
  rsync -a -e "$R" data/xeus_ft/attest.jsonl "root@$IP:/workspace/probe.jsonl" 2>/dev/null
  dt=$(( $(date +%s) - t0 )); rate=$(( 137 / (dt > 0 ? dt : 1) ))
  echo "== probe: 137 MB in ${dt}s = ${rate} MB/s  ($IP)"
  [ "$rate" -ge 4 ] && break
  scripts/xeus_ft_runpod.sh down 2>&1 | tail -1
done

# --- 2. ship: code, text data, 5.4 GB of chunk audio, run-3 clips, run 2 ----
echo "== ship $(date)"
scripts/xeus_ft_runpod.sh ship 2>&1 | grep -v "^\s*$" | tail -3
scripts/xeus_ft_runpod.sh code
$R root@$IP 'mkdir -p /workspace/xeus_ft/data/xeus_ft/ckpt/best /workspace/xeus_ft/data/xeus_ft/seg' 2>/dev/null
rsync -a -e "$R" scripts/xeus_ft_compare.py "root@$IP:/workspace/xeus_ft/scripts/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/run3/segments.jsonl data/xeus_ft/attest.jsonl data/xeus_ft/attest_targets.jsonl "root@$IP:/workspace/xeus_ft/data/xeus_ft/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/run3/seg/ "root@$IP:/workspace/xeus_ft/data/xeus_ft/seg/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/ckpt/best/inner.safetensors data/xeus_ft/ckpt/best/yi_head.pt data/xeus_ft/ckpt/best/meta.json "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt/best/" 2>/dev/null
echo "== shipped $(date)"

# --- 3. the chain, detached on the pod ---------------------------------------
cat > /tmp/ear3_pod.sh <<'POD'
#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/xeus_ft
echo "== B aux frame loss from run 2 $(date)"
python scripts/xeus_ft_train.py --data data/xeus_ft --out data/xeus_ft/ckpt_aux --init-ckpt data/xeus_ft/ckpt/best \
  --epochs 3 --augment --lr-enc 1e-5 --warmup 300 --aux-frame-loss 0.3 --aux-phones "ə,oʊ" --aux-weight 5 2>&1 | grep --line-buffered -v "^Fetching\|note:"
echo "== B compare $(date)"
python scripts/xeus_ft_compare.py --a data/xeus_ft/ckpt/best --b data/xeus_ft/ckpt_aux/last --out data/xeus_ft/compare_aux_last.json 2>&1 | grep -v "^Fetching\|note:"
echo "== A1 attested pretrain $(date)"
python scripts/xeus_ft_train.py --data data/xeus_ft --out data/xeus_ft/ckpt_pre_att --chunks data/xeus_ft/attest_targets.jsonl --attest data/xeus_ft/attest.jsonl --root . \
  --epochs 1 --batch-seconds 90 --lr-enc 2e-5 --warmup 300 --time-budget-min 150 2>&1 | grep --line-buffered -v "^Fetching\|note:\|libmpg123"
echo "== A2 certain clips from A1 $(date)"
python scripts/xeus_ft_train.py --data data/xeus_ft --out data/xeus_ft/ckpt_att --init-ckpt data/xeus_ft/ckpt_pre_att/last \
  --epochs 3 --augment --lr-enc 1.5e-5 --warmup 300 2>&1 | grep --line-buffered -v "^Fetching\|note:"
echo "== A compare $(date)"
python scripts/xeus_ft_compare.py --a data/xeus_ft/ckpt/best --b data/xeus_ft/ckpt_att/best --out data/xeus_ft/compare_att_best.json 2>&1 | grep -v "^Fetching\|note:"
python scripts/xeus_ft_compare.py --a data/xeus_ft/ckpt/best --b data/xeus_ft/ckpt_att/last --out data/xeus_ft/compare_att_last.json 2>&1 | grep -v "^Fetching\|note:"
echo "== EAR3 DONE $(date)"
POD
$R root@$IP 'cat > /workspace/ear3.sh' < /tmp/ear3_pod.sh 2>/dev/null
$R root@$IP 'chmod +x /workspace/ear3.sh; cd /workspace; setsid nohup ./ear3.sh > ear3.log 2>&1 < /dev/null & disown; echo started' 2>/dev/null | grep -v "Warning\|Pseudo"
echo "== launched $(date)"
while true; do
  sleep 300
  line=$($R root@$IP 'grep -aE "^== |^epoch|pretraining on|aux frame|^val_|Traceback|OutOfMemory|^done" /workspace/ear3.log | tail -1' 2>/dev/null | cut -c1-240)
  echo "$(date +%H:%M) $line"
  echo "$line" | grep -q "EAR3 DONE\|Traceback\|OutOfMemory" && break
done

# --- 4. bring everything home (best AND last of every run), then terminate ---
mkdir -p data/xeus_ft/ear3
rsync -a -e "$R" "root@$IP:/workspace/ear3.log" data/xeus_ft/ear3/ 2>/dev/null
rsync -a -e "$R" --include="compare_*.json" --exclude="*" "root@$IP:/workspace/xeus_ft/data/xeus_ft/" data/xeus_ft/ear3/ 2>/dev/null
for c in ckpt_aux ckpt_att ckpt_pre_att; do
  for w in best last; do mkdir -p data/xeus_ft/ear3/$c/$w; rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/$c/$w/" data/xeus_ft/ear3/$c/$w/ 2>/dev/null; done
  rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/$c/train_log.jsonl" data/xeus_ft/ear3/$c/ 2>/dev/null
done
echo "== EAR3 LOCAL DONE $(date)"
scripts/xeus_ft_runpod.sh down 2>&1 | tail -1

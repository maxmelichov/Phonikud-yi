#!/usr/bin/env bash
# Round 3b: the attested curriculum's fine-tune stage again, with the 327 oʊ clips oversampled x8 (docs §27).
# Assumes the round-3 pod is still up with everything shipped (ckpt_pre_att on it).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CLOUD=SECURE
POD_JSON=data/scratch/runpod_pod.json
IP=$(.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['ip'])")
PORT=$(.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['port'])")
R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"
echo "== finishing the round-3 fetch $(date)"
rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_pre_att/last/" data/xeus_ft/ear3/ckpt_pre_att/last/ 2>/dev/null
rsync -a -e "$R" scripts/xeus_ft_train.py "root@$IP:/workspace/xeus_ft/scripts/" 2>/dev/null
cat > /tmp/ear3b_pod.sh <<'POD'
#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/xeus_ft
echo "== A3 certain clips from A1, oʊ x8 $(date)"
python scripts/xeus_ft_train.py --data data/xeus_ft --out data/xeus_ft/ckpt_att_ou --init-ckpt data/xeus_ft/ckpt_pre_att/last \
  --epochs 3 --augment --lr-enc 1.5e-5 --warmup 300 --oversample-phone "oʊ:8" 2>&1 | grep --line-buffered -v "^Fetching\|note:"
for c in best last; do
  python scripts/xeus_ft_compare.py --a data/xeus_ft/ckpt/best --b data/xeus_ft/ckpt_att_ou/$c --out data/xeus_ft/compare_att_ou_$c.json 2>&1 | grep -v "^Fetching\|note:"
  python scripts/xeus_ft_probe_pairs.py --data data/xeus_ft --ckpts data/xeus_ft/ckpt/best data/xeus_ft/ckpt_att_ou/$c --limit-per-pair 600 --out data/xeus_ft/probe_att_ou_$c.json 2>&1 | grep -v "^Fetching\|note:\|clips [0-9]* kept"
done
echo "== EAR3B DONE $(date)"
POD
$R root@$IP 'cat > /workspace/ear3b.sh' < /tmp/ear3b_pod.sh 2>/dev/null
$R root@$IP 'chmod +x /workspace/ear3b.sh; cd /workspace; setsid nohup ./ear3b.sh > ear3b.log 2>&1 < /dev/null & disown; echo started' 2>/dev/null | grep -v "Warning\|Pseudo"
echo "== launched $(date)"
while true; do
  sleep 300
  line=$($R root@$IP 'grep -aE "^== |^epoch|oversampling|^val_|Traceback|OutOfMemory" /workspace/ear3b.log | tail -1' 2>/dev/null | cut -c1-240)
  echo "$(date +%H:%M) $line"
  echo "$line" | grep -q "EAR3B DONE\|Traceback\|OutOfMemory" && break
done
rsync -a -e "$R" "root@$IP:/workspace/ear3b.log" data/xeus_ft/ear3/ 2>/dev/null
rsync -a -e "$R" --include="compare_att_ou_*.json" --include="probe_att_ou_*.json" --exclude="*" "root@$IP:/workspace/xeus_ft/data/xeus_ft/" data/xeus_ft/ear3/ 2>/dev/null
for w in best last; do mkdir -p data/xeus_ft/ear3/ckpt_att_ou/$w; rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_att_ou/$w/" data/xeus_ft/ear3/ckpt_att_ou/$w/ 2>/dev/null; done
echo "== EAR3B LOCAL DONE $(date)"
scripts/xeus_ft_runpod.sh down 2>&1 | tail -1
echo "== EAR3 LOCAL DONE $(date)" >> data/xeus_ft/ear3/driver.log   # releases the v9 chain

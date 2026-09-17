#!/usr/bin/env bash
# Ear round 3c (docs §29): round 3b's fine-tune stage again from ckpt_pre_att/last,
# on the certain clips MINUS the 1,294 whose וי variant contradicts the primary
# reading (scripts/xeus_ft_filter_variants.py -> data/xeus_ft/run3c), oʊ clips ×8.
# Acceptance on the UNFILTERED val clips: compare vs run 2 (PER, p<0.01) and the
# slot probe (oʊ >= 29/62 raw with ɔj >= 480/515); attribution probe 3b vs 3c;
# bias sweep as the calibration fallback.
#   bash scripts/ear_round3c.sh    # create pod -> ship -> train 2 epochs -> evals -> fetch -> down
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CLOUD=SECURE
export RUNPOD_GPUS="${RUNPOD_GPUS:-NVIDIA GeForce RTX 4090,NVIDIA A100-SXM4-80GB,NVIDIA RTX A6000,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090}"
POD_JSON=data/scratch/runpod_pod.json
pod_addr() { .venv/bin/python -c "import json;p=json.load(open('$POD_JSON'));print(p['ip'],p['port'])"; }

for attempt in $(seq 1 12); do
  echo "== create attempt $attempt $(date)"
  rm -f "$POD_JSON"
  scripts/xeus_ft_runpod.sh create 2>&1 | tail -2
  if [ ! -s "$POD_JSON" ]; then
    echo "== no pod (no instances available); retry in 10 min"
    [ "$attempt" -eq 12 ] && { echo "== giving up"; exit 1; }
    sleep 600; continue
  fi
  if ! pod_addr >/dev/null 2>&1; then
    echo "== pod created but never placed; terminating and retrying"
    scripts/xeus_ft_runpod.sh down 2>&1 | tail -1; sleep 120; continue
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

echo "== ship $(date)"
scripts/xeus_ft_runpod.sh ship 2>&1 | grep -v "^\s*$" | tail -3
scripts/xeus_ft_runpod.sh code
$R root@$IP 'mkdir -p /workspace/xeus_ft/data/xeus_ft/ckpt/best /workspace/xeus_ft/data/xeus_ft/ckpt_pre_att/last /workspace/xeus_ft/data/xeus_ft/ckpt_att_ou/best /workspace/xeus_ft/data/xeus_ft/seg /workspace/xeus_ft/data/xeus_ft_c' 2>/dev/null
rsync -a -e "$R" scripts/xeus_ft_compare.py scripts/xeus_ft_probe_pairs.py "root@$IP:/workspace/xeus_ft/scripts/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/run3/segments.jsonl data/xeus_ft/dictionary.json "root@$IP:/workspace/xeus_ft/data/xeus_ft/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/run3c/segments.jsonl "root@$IP:/workspace/xeus_ft/data/xeus_ft_c/segments.jsonl" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/run3/seg/ "root@$IP:/workspace/xeus_ft/data/xeus_ft/seg/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/ckpt/best/inner.safetensors data/xeus_ft/ckpt/best/yi_head.pt data/xeus_ft/ckpt/best/meta.json "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt/best/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/ear3/ckpt_pre_att/last/ "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_pre_att/last/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/ear3/ckpt_att_ou/best/ "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_att_ou/best/" 2>/dev/null
$R root@$IP 'ln -sfn /workspace/xeus_ft/data/xeus_ft/seg /workspace/xeus_ft/data/xeus_ft_c/seg; cp /workspace/xeus_ft/data/xeus_ft/dictionary.json /workspace/xeus_ft/data/xeus_ft_c/; ls /workspace/xeus_ft/data/xeus_ft_c/seg/train | head -1' 2>/dev/null
echo "== shipped $(date)"

cat > /tmp/ear3c_pod.sh <<'POD'
#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/xeus_ft
D=data/xeus_ft
echo "== 3c train from pre_att, filtered clips, oʊ x8 $(date)"
python scripts/xeus_ft_train.py --data data/xeus_ft_c --out $D/ckpt_att_ou_c --init-ckpt $D/ckpt_pre_att/last \
  --epochs 2 --augment --lr-enc 1.5e-5 --warmup 300 --oversample-phone "oʊ:8" 2>&1 | grep --line-buffered -v "^Fetching\|note:"
for c in best last; do
  echo "== compare $c $(date)"
  python scripts/xeus_ft_compare.py --data $D --a $D/ckpt/best --b $D/ckpt_att_ou_c/$c --out $D/compare_att_ou_c_$c.json 2>&1 | grep -v "^Fetching\|note:"
  echo "== probe $c $(date)"
  python scripts/xeus_ft_probe_pairs.py --data $D --ckpts $D/ckpt/best $D/ckpt_att_ou_c/$c --limit-per-pair 600 --out $D/probe_att_ou_c_$c.json 2>&1 | grep -v "^Fetching\|note:\|clips [0-9]* kept"
done
echo "== attribution 3b vs 3c $(date)"
python scripts/xeus_ft_probe_pairs.py --data $D --ckpts $D/ckpt_att_ou/best $D/ckpt_att_ou_c/best --limit-per-pair 600 --out $D/probe_3b_vs_3c.json 2>&1 | grep -v "^Fetching\|note:\|clips [0-9]* kept"
echo "== bias sweep $(date)"
for B in 0.5 1 1.5 2; do
  python scripts/xeus_ft_probe_pairs.py --data $D --ckpts $D/ckpt_att_ou_c/best --pairs "oʊ:ɔj" --splits val_words --phone-bias oʊ:$B --out $D/probe_att_ou_c_bias_$B.json 2>&1 | grep "true reading wins"
done
echo "== EAR3C DONE $(date)"
POD
$R root@$IP 'cat > /workspace/ear3c.sh' < /tmp/ear3c_pod.sh 2>/dev/null
$R root@$IP 'chmod +x /workspace/ear3c.sh; cd /workspace; setsid nohup ./ear3c.sh > ear3c.log 2>&1 < /dev/null & disown; echo started' 2>/dev/null | grep -v "Warning\|Pseudo"
echo "== launched $(date)"
while true; do
  sleep 300
  line=$($R root@$IP 'grep -aE "^== |^epoch|oversampling|^val_|Traceback|OutOfMemory" /workspace/ear3c.log | tail -1' 2>/dev/null | cut -c1-240)
  echo "$(date +%H:%M) $line"
  echo "$line" | grep -q "EAR3C DONE\|Traceback\|OutOfMemory" && break
done
mkdir -p data/xeus_ft/ear3c
rsync -a -e "$R" "root@$IP:/workspace/ear3c.log" data/xeus_ft/ear3c/ 2>/dev/null
rsync -a -e "$R" --include="compare_att_ou_c_*.json" --include="probe_att_ou_c*.json" --include="probe_3b_vs_3c.json" --exclude="*" "root@$IP:/workspace/xeus_ft/data/xeus_ft/" data/xeus_ft/ear3c/ 2>/dev/null
for w in best last; do mkdir -p data/xeus_ft/ear3c/ckpt_att_ou_c/$w; rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_att_ou_c/$w/" data/xeus_ft/ear3c/ckpt_att_ou_c/$w/ 2>/dev/null; done
rsync -a -e "$R" "root@$IP:/workspace/xeus_ft/data/xeus_ft/ckpt_att_ou_c/train_log.jsonl" data/xeus_ft/ear3c/ckpt_att_ou_c/ 2>/dev/null
echo "== EAR3C LOCAL DONE $(date)"
scripts/xeus_ft_runpod.sh down 2>&1 | tail -1

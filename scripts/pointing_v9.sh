#!/usr/bin/env bash
# Pointing model v9 = the v8 recipe on retrain9 (attest_v9.jsonl: the ear's
# decisions gated and lifted by ReNikud-yi, scripts/renikud_attest.py).
#   bash scripts/pointing_v9.sh          # pod -> ship -> train 2 epochs from v6 -> fetch -> down
# Then locally: export ONNX + metadata, paired audio yardstick vs v8 (docs §28).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CLOUD=SECURE
POD_JSON=data/scratch/runpod_pod.json
pod_addr() { .venv/bin/python -c "import json;p=json.load(open('$POD_JSON'));print(p['ip'],p['port'])"; }

for attempt in $(seq 1 18); do
  echo "== create attempt $attempt $(date)"
  rm -f "$POD_JSON"
  scripts/xeus_ft_runpod.sh create 2>&1 | tail -2
  if [ ! -s "$POD_JSON" ]; then
    echo "== no pod; retry in 10 min"; [ "$attempt" -eq 18 ] && { echo "== giving up"; exit 1; }
    sleep 600; continue
  fi
  read -r IP PORT < <(pod_addr)
  R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"
  break
done

echo "== ship $(date)"
$R root@$IP 'mkdir -p /workspace/v9/scripts /workspace/v9/phonikud /workspace/v9/models/phonikud_yi_v6 /workspace/v9/data && pip install -q "transformers==4.56.2" safetensors 2>&1 | tail -1' 2>/dev/null
rsync -a -e "$R" scripts/train_phonikud_yi.py scripts/phonikud_yi_data.py "root@$IP:/workspace/v9/scripts/" 2>/dev/null
rsync -a -e "$R" --exclude __pycache__ phonikud/model "root@$IP:/workspace/v9/phonikud/" 2>/dev/null
rsync -a -e "$R" data/retrain9 "root@$IP:/workspace/v9/data/" 2>/dev/null
rsync -a -e "$R" models/phonikud_yi_v6/best "root@$IP:/workspace/v9/models/phonikud_yi_v6/" 2>/dev/null
echo "== shipped $(date)"

cat > /tmp/v9_pod.sh <<'POD'
#!/usr/bin/env bash
cd /workspace/v9
echo "== train v9 $(date)"
python scripts/train_phonikud_yi.py --data data/retrain9 --init models/phonikud_yi_v6/best --out models/phonikud_yi_v9 \
  --epochs 2 --batch-size 8 --lr 5e-6 --head-lr 5e-5 2>&1 | grep --line-buffered -v "^Fetching\|note:"
echo "== V9 DONE $(date)"
POD
$R root@$IP 'cat > /workspace/v9.sh' < /tmp/v9_pod.sh 2>/dev/null
$R root@$IP 'chmod +x /workspace/v9.sh; cd /workspace; setsid nohup ./v9.sh > v9.log 2>&1 < /dev/null & disown; echo started' 2>/dev/null | grep -v "Warning\|Pseudo"
echo "== launched $(date)"
while true; do
  sleep 300
  line=$($R root@$IP 'grep -aE "^== |step [0-9]*/|^  val|^  e[0-9] val|Traceback|OutOfMemory|best" /workspace/v9.log | tail -1' 2>/dev/null | cut -c1-200)
  echo "$(date +%H:%M) $line"
  echo "$line" | grep -q "V9 DONE\|Traceback\|OutOfMemory" && break
done
mkdir -p models/phonikud_yi_v9
rsync -a -e "$R" "root@$IP:/workspace/v9.log" models/phonikud_yi_v9/train_v9.log 2>/dev/null
rsync -a -e "$R" "root@$IP:/workspace/v9/models/phonikud_yi_v9/" models/phonikud_yi_v9/ 2>/dev/null
echo "== V9 LOCAL DONE $(date)"
scripts/xeus_ft_runpod.sh down 2>&1 | tail -1

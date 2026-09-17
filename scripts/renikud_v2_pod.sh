#!/usr/bin/env bash
# ReNikud-yi v2: retrain the per-letter context reader on the NEW ear's labels
# (stress included) and measure it, paired against the shipped v1, on a pod.
#
#   1. locally: data/renikud_yi_v2 must exist (scripts/renikud_yi_prepare_v2.py on the REAL
#      xeus-yi-ipa/data/attest_lattice.jsonl — the script refuses a synthetic set), and the
#      three attestation files are cut down to the six held-out episodes for the eval.
#   2. pod: create with retries + link probe (style of ear_round3c.sh), ship scripts + data +
#      the v6 pointing body (init) + the shipped v1 model (the paired baseline).
#   3. train: renikud_yi_train.py with the PRODUCTION hyper-parameters of models/renikud_yi_audio
#      (train.py defaults: 3 epochs, batch 8, max-chars 480, encoder lr 2e-5, head lr 1e-4,
#      warmup 6%, weight decay 0.01, seed 17, bf16 autocast; 11,940 steps / 97 min on an A6000
#      for v1's 31,835 chunks) from models/phonikud_yi_v6/best.
#   4. eval (renikud_yi_eval_v2.py), v1 and v2 paired, on the six held-out episodes' rule-path
#      words with a decision at margin >= 2:
#        (i)  against the NEW ear (attest_lattice.jsonl): segments + stress
#        (ii) against the OLD ear (attest.jsonl): segments (v1's own yardstick, 94.5%)
#   5. fetch models/renikud_yi_v2/{best,last_heads.pt,train_log.jsonl} + eval json, terminate.
#
# Acceptance (one line): on (i), v2+graph vs v1+graph paired sign test on the rule-path
# words: net > 0 with p < 0.01, AND stress_acc of v2+graph above v1+graph+engine-stress
# (p < 0.01 on the polysyllables); (ii) must not regress (p >= 0.01 or net >= 0).
#
#   bash scripts/renikud_v2_pod.sh            # everything
#   STAGE=local bash scripts/renikud_v2_pod.sh   # only the local staging + checks
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CLOUD=SECURE
export RUNPOD_GPUS="${RUNPOD_GPUS:-NVIDIA RTX A6000,NVIDIA A100-SXM4-80GB,NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090}"
PY=.venv/bin/python
POD_JSON=data/scratch/runpod_pod.json
XEUS_YI=../xeus-yi-ipa
DATA="${DATA:-data/renikud_yi_v2}"
OUT="${OUT:-models/renikud_yi_v2}"
EPOCHS="${EPOCHS:-3}"
HELD=100313,104192,104690,113370,58622,94226
SHIP=data/scratch/renikud_v2_ship
REMOTE=/workspace/renikud
pod_addr() { $PY -c "import json;p=json.load(open('$POD_JSON'));print(p['ip'],p['port'])"; }

# ---------------------------------------------------------------- local staging
echo "== local checks $(date)"
[ -s "$DATA/train.jsonl" ] || { echo "no $DATA/train.jsonl: run scripts/renikud_yi_prepare_v2.py first"; exit 1; }
[ -s "$XEUS_YI/data/attest_lattice.jsonl" ] || { echo "no $XEUS_YI/data/attest_lattice.jsonl (the sibling stream's output)"; exit 1; }
$PY - "$DATA" <<'EOF' || exit 1
import json, sys
cfg = json.load(open(sys.argv[1] + "/prepare_config.json"))
if cfg.get("attest_synthetic"):
    sys.exit(f"{sys.argv[1]} was built from a SYNTHETIC attest file ({cfg['attest']}); rebuild on the real one")
held = {"100313", "104192", "104690", "113370", "58622", "94226"}
for split in ("train", "val"):
    eps = {json.loads(l)["episode"] for l in open(f"{sys.argv[1]}/{split}.jsonl", encoding="utf-8")}
    leak = (eps & held) | (eps & set(cfg["val_eps"]))
    if leak:
        sys.exit(f"{split}.jsonl contains excluded episodes: {sorted(leak)}")
print("data ok:", cfg["attest"], "trainer val", cfg["trainer_val"])
EOF
mkdir -p "$SHIP"
$PY - "$SHIP" "$XEUS_YI" <<'EOF'
import json, sys
ship, xy = sys.argv[1], sys.argv[2]
held = {"100313", "104192", "104690", "113370", "58622", "94226"}
for src, dst in ((f"{xy}/data/attest_lattice.jsonl", "attest_lattice_heldout.jsonl"),
                 ("data/xeus_ft/attest.jsonl", "attest_old_heldout.jsonl"),
                 ("data/xeus_ft/attest_targets.jsonl", "attest_targets_heldout.jsonl")):
    n = 0
    with open(f"{ship}/{dst}", "w", encoding="utf-8") as fh:
        for line in open(src, encoding="utf-8"):
            if json.loads(line)["episode"] in held:
                fh.write(line); n += 1
    print(f"{dst}: {n:,} rows")
EOF
$PY scripts/renikud_yi_prepare_v2.py --check-only --attest "$XEUS_YI/data/attest_lattice.jsonl" --check-rows 2000 || exit 1
[ "${STAGE:-}" = "local" ] && { echo "== local staging done"; exit 0; }

# ---------------------------------------------------------------- pod
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
  R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -p $PORT"
  t0=$(date +%s)
  rsync -a -e "$R" data/corpus/yiddish_tts_dataset.tsv "root@$IP:/workspace/probe.tsv" 2>/dev/null
  dt=$(( $(date +%s) - t0 )); rate=$(( 62 / (dt > 0 ? dt : 1) ))
  echo "== probe: 62 MB in ${dt}s = ${rate} MB/s  ($IP)"
  [ "$rate" -ge 4 ] && break
  scripts/xeus_ft_runpod.sh down 2>&1 | tail -1
done

echo "== ship $(date)"
$R root@$IP "mkdir -p $REMOTE/scripts $REMOTE/data/corpus $REMOTE/data/xeus_ft $REMOTE/data/eval $REMOTE/models/phonikud_yi_v6/best $REMOTE/models/renikud_yi_audio $REMOTE/$DATA && \
  pip install -q 'transformers==4.56.2' safetensors 2>&1 | tail -1; python -c 'import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))'" 2>/dev/null
rsync -a -e "$R" scripts/renikud_yi_train.py scripts/renikud_yi_eval_v2.py scripts/renikud_yi_prepare_v2.py scripts/yi_align.py scripts/xeus_ft_common.py scripts/xeus_lattice.py "root@$IP:$REMOTE/scripts/" 2>/dev/null
rsync -a -e "$R" "$DATA/" "root@$IP:$REMOTE/$DATA/" 2>/dev/null
rsync -a -e "$R" data/corpus/yiddish_tts_dataset.tsv "root@$IP:$REMOTE/data/corpus/" 2>/dev/null
rsync -a -e "$R" data/xeus_ft/dictionary.json "$SHIP/attest_lattice_heldout.jsonl" "$SHIP/attest_old_heldout.jsonl" "$SHIP/attest_targets_heldout.jsonl" "root@$IP:$REMOTE/data/xeus_ft/" 2>/dev/null
rsync -a -e "$R" --exclude='__pycache__' models/phonikud_yi_v6/best/ "root@$IP:$REMOTE/models/phonikud_yi_v6/best/" 2>/dev/null
rsync -a -e "$R" models/renikud_yi_audio/heads.pt models/renikud_yi_audio/best_encoder "root@$IP:$REMOTE/models/renikud_yi_audio/" 2>/dev/null
echo "== shipped $(date)"

cat > /tmp/renikud_v2_pod.sh <<POD
#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REMOTE
echo "== train v2 \$(date)"
python scripts/renikud_yi_train.py --data $DATA --init models/phonikud_yi_v6/best --out $OUT --epochs $EPOCHS 2>&1 | grep --line-buffered -v "^Fetching\|note:"
for REF in lattice old; do
  echo "== eval vs \$REF ear \$(date)"
  python scripts/renikud_yi_eval_v2.py --ref data/xeus_ft/attest_\${REF}_heldout.jsonl --targets data/xeus_ft/attest_targets_heldout.jsonl \\
    --episodes $HELD --margin 2 --models models/renikud_yi_audio $OUT --names v1 v2 \\
    --out data/eval/renikud_v2_vs_\${REF}_ear.json 2>&1 | grep -v "^Fetching\|note:\|Warning"
done
echo "== RENIKUD V2 DONE \$(date)"
POD
$R root@$IP "cat > $REMOTE/run.sh" < /tmp/renikud_v2_pod.sh 2>/dev/null
$R root@$IP "chmod +x $REMOTE/run.sh; cd $REMOTE; setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown; echo started" 2>/dev/null | grep -v "Warning\|Pseudo"
echo "== launched $(date)"
while true; do
  sleep 300
  line=$($R root@$IP "grep -aE '^== |^epoch|new best|Traceback|OutOfMemory|rule-path words' $REMOTE/run.log | tail -1" 2>/dev/null | cut -c1-240)
  echo "$(date +%H:%M) $line"
  echo "$line" | grep -q "RENIKUD V2 DONE\|Traceback\|OutOfMemory" && break
done
mkdir -p "$OUT" data/eval
rsync -a -e "$R" "root@$IP:$REMOTE/run.log" "$OUT/train_renikud_yi_v2.log" 2>/dev/null
for f in train_log.jsonl last_heads.pt; do rsync -a -e "$R" "root@$IP:$REMOTE/$OUT/$f" "$OUT/" 2>/dev/null; done
rsync -a -e "$R" "root@$IP:$REMOTE/$OUT/best/" "$OUT/best/" 2>/dev/null
rsync -a -e "$R" --include='renikud_v2_vs_*' --exclude='*' "root@$IP:$REMOTE/data/eval/" data/eval/ 2>/dev/null
echo "== RENIKUD V2 LOCAL DONE $(date): $OUT/best, data/eval/renikud_v2_vs_{lattice,old}_ear.json"
scripts/xeus_ft_runpod.sh down 2>&1 | tail -1

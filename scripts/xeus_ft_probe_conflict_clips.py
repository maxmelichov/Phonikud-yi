"""Score the 1,294 train clips proposal 1 would drop (two-variant word, aligner chose the ɔj variant)
and the 300 kept oʊ clips: does each ear prefer oʊ or ɔj on those slots?"""
import json, random, sys, collections
from pathlib import Path
REPO = Path('/Users/maxm/Documents/yiddish/Phonikud-yi')
sys.path.insert(0, str(REPO / 'scripts'))
import torch
from xeus_ft_common import read_jsonl, yi_logits
from xeus_ft_train import Segments, collate
from xeus_yi_decode import load_finetuned
from xeus_ft_probe_pairs import nll
two = {'אויכ','אויס','ארויס','לויט','הויז','טוישנ'}
rows = []
for r in read_jsonl(REPO/'data/xeus_ft/run3/segments.jsonl'):
    if r['split'] != 'train': continue
    t = r['target']
    bad = [w['key'] for w in r['words'] if w['key'] in two and w.get('variant',0) != 0]
    if bad and t.count('ɔj') == 1 and 'oʊ' not in t:
        rows.append({**r, '_grp': 'drop:'+bad[0], '_true': 'ɔj', '_swap': 'oʊ'})
    elif not bad and t.count('oʊ') == 1 and 'ɔj' not in t:
        k = [w['key'] for w in r['words'] if w['key'] in two]
        rows.append({**r, '_grp': 'keep:'+(k[0] if k else '?'), '_true': 'oʊ', '_swap': 'ɔj'})
print(collections.Counter(r['_grp'] for r in rows), flush=True)
ds = Segments(rows, REPO/'data/xeus_ft/run3/seg')
out = {}
for ck in sys.argv[1:]:
    inner, head = load_finetuned(Path(ck), 'mps'); inner.eval(); head.eval()
    w = [None]*len(rows)
    with torch.no_grad():
        for idx in ds.batches(40.0, shuffle=False, rng=random.Random(0)):
            speech, lens, _, _ = collate(ds, idx, 'mps')
            logits, flens = yi_logits(inner, head, speech, lens)
            lp = torch.log_softmax(logits.float(), -1).cpu()
            for j, i in enumerate(idx):
                r = rows[i]
                sw = [r['_swap'] if p == r['_true'] else p for p in r['target']]
                t, s = nll(lp[j, :int(flens[j])], [r['target'], sw])
                w[i] = s - t
    summ = {}
    for r, m in zip(rows, w):
        g = summ.setdefault(r['_grp'], [0, 0, []]); g[0 if m > 0 else 1] += 1; g[2].append(m)
    out[ck] = {g: {'label_wins': v[0], 'other_wins': v[1], 'median_margin': round(sorted(v[2])[len(v[2])//2], 2)} for g, v in summ.items()}
    print(ck, json.dumps(out[ck], ensure_ascii=False), flush=True)
    del inner, head
json.dump(out, open(Path(__file__).parent/'probe_conflict.json', 'w'), ensure_ascii=False, indent=1)

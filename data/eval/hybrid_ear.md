# Decode-time hybrid ear: run 2 for the וי slot, ckpt_att_ou for the rest (2026-09-17)

Script `scripts/xeus_hybrid_eval.py`; numbers `data/xeus_ft/ear3/hybrid_eval.json`;
cached per-candidate NLLs `data/xeus_ft/ear3/hybrid_nll_{A,B}.jsonl`; per-clip
decisions `data/xeus_ft/ear3/hybrid_decisions.jsonl`. Ear A = run 2
(`data/xeus_ft/ckpt/best`), ear B = `data/xeus_ft/ear3/ckpt_att_ou/best`.
Encoders on MPS, `ctc_loss` on CPU; 1:38 min per ear for 3,017 clips.

**Clips.** `data/xeus_ft/run3`, seeded sample of 1,500 clips per split plus every
clip whose target holds oʊ: val_words 1,530 clips / 7,167 words (70 oʊ clips),
val_eps 1,487 clips / 4,673 words (10 oʊ clips). 3 sampled clips dropped (word
not in `dictionary.json`).

**Candidates.** Per word, `xeus_lattice.graph_candidates` around the clip's gold
reading (≤96, mean 5.4 per word in val_words); each candidate is scored as the
whole clip with the other words held at gold, i.e. one coordinate step of
`xeus_lattice.choose` from the gold initialisation. Every ear and hybrid
re-ranks the identical list, so the combination is fully offline. Because the
neighbours are held at gold, the absolute exact-match rates are higher than the
lattice's graph level in §12; only the paired differences matter here. Slot
accuracy is over *contested* slots (positions where some candidate differs).
`zero_infinity=False` (an impossible candidate scores +inf, not 0); no
candidate was infinite.

**Decoders.** A, B = argmin NLL. H1 = B's argmin, then A's argmin among the
candidates that differ from it only in וי slots. H2 = A picks within every וי
class first, B picks among the class representatives. S(λ) = argmin
NLL_B + λ·(NLL_A − min over the candidate's וי class of NLL_A), λ ∈ {0.5, 1}.
H1 and H2 made identical choices on every word (the וי class is almost always
binary).

## Results

val_words (unseen word types), n = 1,530 clips / 7,167 words:

| | A (run 2) | B (att_ou) | H1 = H2 | S0.5 | S1 |
|---|---|---|---|---|---|
| clip exact | 0.446 | **0.526** | 0.525 | **0.528** | 0.528 |
| word exact | 0.808 | 0.841 | 0.840 | 0.841 | 0.841 |
| slot oʊ (n=71) | **0.451** | 0.310 | 0.451 | 0.366 | 0.422 |
| slot ɔj (n=314) | 0.943 | **0.987** | 0.943 | 0.984 | 0.971 |
| slot ə (n=1,702) | 0.797 | 0.870 | 0.870 | 0.870 | 0.870 |
| slot ɛ (n=1,524) | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 |
| slot a/ɔ/u (n=2,792) | 0.776 | 0.798 | 0.798 | 0.798 | 0.798 |
| slot i/u (n=3,390) | 0.843 | 0.866 | 0.866 | 0.866 | 0.866 |
| slot aj/ej (n=826) | 0.875 | 0.886 | 0.886 | 0.886 | 0.886 |
| slot f/p (n=517) | 0.836 | 0.911 | 0.911 | 0.911 | 0.911 |

val_eps (unseen episodes), n = 1,487 clips / 4,673 words:

| | A | B | H1 = H2 | S0.5 | S1 |
|---|---|---|---|---|---|
| clip exact | 0.631 | **0.650** | 0.645 | **0.650** | 0.649 |
| word exact | 0.838 | 0.844 | 0.842 | 0.844 | 0.843 |
| slot oʊ (n=10) | 0.800 | 0.700 | 0.800 | 0.800 | 0.700 |
| slot ɔj (n=204) | 0.922 | **0.971** | 0.922 | 0.961 | 0.961 |
| slot ə (n=928) | 0.865 | 0.868 | 0.868 | 0.868 | 0.868 |
| slot ɛ (n=1,015) | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 |
| slot a/ɔ/u (n=1,823) | 0.804 | 0.811 | 0.811 | 0.811 | 0.811 |
| slot i/u (n=2,135) | 0.841 | 0.857 | 0.857 | 0.857 | 0.857 |

Paired sign tests (first-only / second-only, p):

| | val_words clip | val_words word | val_eps clip | val_eps word |
|---|---|---|---|---|
| A vs B | 42 / 165, 2e-18 | 169 / 401, 1e-22 | 68 / 96, 0.035 | 132 / 157, 0.16 |
| A vs H1 | 39 / 160, 1e-18 | 163 / 389, 3e-22 | 68 / 88, 0.13 | 130 / 146, 0.37 |
| B vs H1 | 6 / 4, 0.75 | 13 / 7, 0.26 | **8 / 0, 0.008** | **11 / 2, 0.02** |
| B vs S0.5 | 0 / 3, 0.25 | 1 / 4, 0.38 | 1 / 0, 1 | 2 / 1, 1 |
| B vs S1 | 1 / 4, 0.38 | 5 / 7, 0.77 | 2 / 0, 0.5 | 3 / 1, 0.6 |

Slot-level, val_words: oʊ B vs H1 0 / 10 (p = 0.002) — the hybrid recovers
every oʊ that run 2 alone gets and B misses; ɔj B vs H1 14 / 0 (p = 1e-4) —
and loses exactly as many ɔj. val_eps: ɔj B vs H1 10 / 0 (p = 0.002),
oʊ 1 / 2.

## What it says

1. **The hard hybrid is a wash on unseen words and a small loss on unseen
   episodes.** The וי slot is binary. Run 2 says "oʊ" more often than B, so
   handing it the slot buys back the 10 oʊ tokens B misses (71 contested oʊ
   slots) and costs 14 of the 314 ɔj tokens B had right — net −4 words, and
   on val_eps −10 ɔj for +2 oʊ. Run 2's "oʊ advantage" in the §27b probe is a
   bias toward oʊ, not better discrimination: on the 71 val_words oʊ slots it
   is right on 32 and B on 22, but among the 314 ɔj slots it is wrong on 18 and
   B on 4. Per token of וי, B is the better ear (val_words 336/385 vs 328/385).
2. **The soft hybrid S0.5 is the best decoder by 3 clips on val_words and 0-1
   on val_eps**, nowhere near significance (p ≥ 0.25). Not worth a second
   resident ear.
3. **The val_words oʊ probe is one word.** 65 of the 71 contested oʊ slots are
   ארויפ (aroʊf); the rest are ארויס ×3, הויז ×2, אויכ ×1. On ארויפ A is right
   26/65, B 17/65, both wrong on 39. Every "oʊ on unseen words" number in
   §27-27b (the 62-clip probe included) is a measurement of this one type.
   val_eps has 10 oʊ slots over 5 types. The oʊ canary cannot be judged on this
   validation set; it needs more oʊ word types held out, or a Chezky-labelled
   oʊ test list.
4. B (`ckpt_att_ou/best`) alone beats run 2 on every contested slot except oʊ
   in both splits (ə 0.797 → 0.870 on unseen words, f/p 0.836 → 0.911, clip
   exact 0.446 → 0.526, p = 2e-18; val_eps clip exact 0.631 → 0.650,
   p = 0.035). The decision of §27b — run 2 stays for anything touching
   Chezky's labels — is unaffected by this measurement; but the case for a
   hybrid is closed: if B is adopted for a task, adopt it whole.

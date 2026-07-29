#!/usr/bin/env python3
"""
Assemble models/phonikud_yi_small/report.{md,json} from the pieces produced by
training, evaluation and benchmarking.

Usage:
    python scripts/make_small_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SMALL = REPO / "models/phonikud_yi_small"
R4 = REPO / "models/phonikud_yi/round4"


def jload(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    teacher_eval = jload(R4 / "oov_wordlevel_report.json")
    student_eval = jload(SMALL / "oov_wordlevel_student.json")
    student_int8_eval = jload(SMALL / "oov_wordlevel_student_int8.json")
    bench = jload(SMALL / "benchmark.json", {})

    val = {
        "teacher (round 4, 306M)": jload(R4 / "stageB/metrics.json", {}).get("char_acc"),
        "student gold-only (18.4M)": jload(SMALL / "gold_only/best_metrics.json", {}).get("char_acc"),
        "student gold+pseudo (18.4M)": jload(SMALL / "student_kd/best_metrics.json", {}).get("char_acc"),
    }

    out = {
        "val_char_acc": val,
        "test": {
            "teacher": teacher_eval,
            "student": student_eval,
            "student_int8": student_int8_eval,
        },
        "benchmark": bench,
    }
    L = ["# Small Yiddish diacritizer — distilled student\n"]
    L.append("A from-scratch character-level ModernBERT-style encoder (RoPE, GeGLU, "
             "bias-free, pre-LN), **18,416,666 parameters**, 7 layers x 512 dim x 8 heads, "
             "FFN 1024, 88-symbol char vocabulary, with the same three Yiddish heads as "
             "round 4 (22-way vowel x dagesh, 2-way shin/sin, binary rafe).\n")
    L.append("\nTrained on canonical gold text plus round-4 teacher pseudo-labels over the "
             "11,662 annotation chunks the alignment filter had rejected, with the canonical "
             "pointing map applied on top of the teacher output (the map wins). Hard labels.\n")

    # ---- ablation -------------------------------------------------------
    L.append("\n## Does the teacher data help? (val, all-char accuracy)\n")
    L.append("| model | params | val char acc |")
    L.append("|---|---:|---:|")
    for k, v in val.items():
        par = "306M" if "round 4" in k else "18.4M"
        L.append(f"| {k.split(' (')[0]} | {par} | {v:.2f}% |" if v is not None else f"| {k} | {par} | n/a |")
    g, gp = val["student gold-only (18.4M)"], val["student gold+pseudo (18.4M)"]
    if g and gp:
        L.append(f"\nAdding the teacher pseudo-labels is worth **+{gp-g:.2f}** points "
                 f"({g:.2f}% -> {gp:.2f}%), which is why the shipped student uses them.\n")

    # ---- accuracy table -------------------------------------------------
    if teacher_eval and student_eval:
        L.append("\n## Test-set accuracy: student vs round-4 teacher\n")
        L.append("Canonical test split, 13 held-out episodes. A character counts correct only "
                 "if nikud, shin/sin dot and rafe are all right; a word counts correct only if "
                 "every character in it is right.\n")
        rows = [("all-char accuracy", "overall", "char_acc"),
                ("marked-char accuracy", "overall", "marked_char_acc"),
                ("word-level accuracy", "overall", "word_acc"),
                ("word-level, >=4 chars", "overall", "word_acc_ge4"),
                ("in-vocab char accuracy", "in_vocab", "char_acc"),
                ("in-vocab word accuracy", "in_vocab", "word_acc"),
                ("**OOV char accuracy**", "oov", "char_acc"),
                ("**OOV marked-char accuracy**", "oov", "marked_char_acc"),
                ("**OOV word accuracy**", "oov", "word_acc")]
        L.append("| metric | teacher (306M) | student (18.4M) | delta |")
        L.append("|---|---:|---:|---:|")
        for label, grp, key in rows:
            a = teacher_eval[grp][key]
            b = student_eval[grp][key]
            L.append(f"| {label} | {a:.2f}% | {b:.2f}% | {b-a:+.2f} |")

        iv = student_eval["in_vocab"]["char_acc"] - teacher_eval["in_vocab"]["char_acc"]
        L.append(f"\nGroup sizes are identical for both models: "
                 f"{student_eval['in_vocab']['n_words']:,} in-vocab and "
                 f"{student_eval['oov']['n_words']:,} OOV word tokens "
                 f"({student_eval['oov_word_rate']:.1f}% OOV).\n")
        L.append(f"\nIn-vocab is the target the brief set (within 1-2 points): the student is "
                 f"**{iv:+.2f}** points on in-vocab characters.\n")

        if student_int8_eval:
            L.append(f"\nInt8 quantisation costs "
                     f"{student_int8_eval['overall']['char_acc']-student_eval['overall']['char_acc']:+.2f} "
                     f"points of all-char accuracy "
                     f"({student_int8_eval['overall']['char_acc']:.2f}% vs "
                     f"{student_eval['overall']['char_acc']:.2f}%).\n")

    # ---- benchmark ------------------------------------------------------
    if bench.get("models"):
        L.append(f"\n## CPU speed on this Mac ({bench.get('threads')} threads, "
                 f"{bench['n_lines']} lines / {bench['n_chars']:,} chars)\n")
        bkeys = sorted({k for m in bench["models"].values() for k in m["runs"]})
        L.append("| model | file size | params | " +
                 " | ".join(f"chars/s {b}" for b in bkeys) + " |")
        L.append("|---|---:|---:|" + "---:|" * len(bkeys))
        for name, m in bench["models"].items():
            cells = [str(m["runs"][b]["chars_per_sec"]) if b in m["runs"] else "-" for b in bkeys]
            L.append(f"| {name} | {m['file_mb']:.1f} MB | {m['params']/1e6:.1f}M | "
                     + " | ".join(cells) + " |")

        try:
            s = bench["models"]["student ONNX int8"]["runs"]["batch1"]["chars_per_sec"]
            t = bench["models"]["teacher torch fp32"]["runs"]["batch1"]["chars_per_sec"]
            sm = bench["models"]["student ONNX int8"]["file_mb"]
            tm = bench["models"]["teacher torch fp32"]["file_mb"]
            L.append(f"\nThe shipped int8 student is **{s/t:.1f}x faster** than the torch "
                     f"teacher at batch 1 and **{tm/sm:.0f}x smaller** on disk "
                     f"({sm:.1f} MB vs {tm:.0f} MB).\n")
        except KeyError:
            pass

    # ---- dictionary-first shipped pipeline ------------------------------
    df = jload(SMALL / "dict_first.json")
    if df:
        out["dict_first"] = df
        L.append("\n## The shipped pipeline (dictionary-first)\n")
        L.append("| configuration | word accuracy |")
        L.append("|---|---:|")
        L.append(f"| model alone (`--no-dict`) | {df['word_acc_model_only']:.2f}% |")
        L.append(f"| dictionary-first (default) | {df['word_acc_dict_first']:.2f}% |")
        L.append(f"\nThe dictionary covers **{df['dict_coverage_pct']:.1f}%** of test word "
                 f"tokens and is {df['dict_lookup_word_acc']:.0f}% accurate on them.\n")
        L.append("\n> **Read that 100% with care — it is circular.** The canonical map was "
                 "derived from the training split and then applied to *all* splits, so the test "
                 "targets for mapped words are by construction exactly what the map returns. "
                 "The dictionary-first figure therefore measures dictionary *coverage*, not "
                 "generalization. The number that reflects genuine generalization is the OOV "
                 "row in the table above, and the honest reading of the product is: near-perfect "
                 "on the ~94% of running text already in the lexicon, and weak on the rest.\n")

    # ---- provenance -----------------------------------------------------
    L.append("\n## Notes and provenance\n")
    L.append("- **Architecture:** ModernBERT ingredients used are pre-LN, RoPE, GeGLU and "
             "bias-free linears. ModernBERT's alternating local/global attention is *not* used "
             "— every layer attends globally. The sliding-window mask is the one component that "
             "traces badly to ONNX, and at 512 characters a local window covers most of the "
             "sequence anyway.\n")
    L.append("- **Learning rates** are the prescribed from-scratch recipe (encoder 2e-5, heads "
             "1e-4). I probed these first because 2e-5 looked low for a randomly initialised "
             "encoder; a 1500-step check reached 89.3% val, so they were kept unchanged.\n")
    L.append("- **Distillation is hard-label.** The teacher's argmax pointing is used as text, "
             "with the canonical map overriding it wherever a word is in the map (167,499 of "
             "883,426 pseudo-label words were overridden). No soft-label KD.\n")
    L.append("- **A label-corruption bug was found and fixed before the final run.** Two "
             "defects compounded: (1) `text_yi` carries partial nikud on 7.7% of words and was "
             "fed to the teacher unstripped, so the teacher saw out-of-distribution pointed "
             "input and stacked its own marks on top; (2) a `split_token` helper treated a "
             "word-final combining mark as trailing punctuation, so a canonical replacement got "
             "the stolen mark re-appended (זִיךְ → זִיךְְ). Together these left **12.7% of "
             "pseudo-label words carrying doubled marks.** Both are fixed, the pseudo-labels "
             "were regenerated from stripped input (now 0 doubled marks, 0 letters with two "
             "vowels), and the student was retrained. The same `split_token` bug was live in the "
             "shipping inference wrapper and is fixed there too.\n")
    L.append("- **int8 is smaller but not faster on this Mac** (18.9 MB vs 74.0 MB, 13.7k vs "
             "14.3k chars/s). Dynamic quantisation adds dequantisation overhead that outweighs "
             "the cheaper matmuls at this model size on Apple silicon. Ship int8 for size; there "
             "is no speed argument for it here.\n")
    L.append("- **Batch 8 is slower than batch 1** for every model, because batching pads to the "
             "longest line in the batch and the padding is wasted compute. Length-bucketed "
             "batching would fix that if throughput ever matters.\n")

    L.append("\n## Files\n")
    for f, what in [
        ("student.onnx", "fp32 ONNX, vocab + class tables embedded in metadata"),
        ("student.int8.onnx", "dynamic-int8 ONNX (the one to ship)"),
        ("student_kd/", "torch checkpoint, config and char vocab (gold+pseudo)"),
        ("gold_only/", "gold-only ablation checkpoint"),
        ("report.md / report.json", "this report"),
        ("benchmark.json", "raw benchmark numbers"),
        ("dict_first.json", "dictionary-first vs model-alone word accuracy"),
        ("oov_wordlevel_student.md/.json", "full student eval incl. OOV examples"),
        ("logs/", "training logs for both students"),
    ]:
        L.append(f"- `{f}` — {what}")

    L.append("\n## Inference\n")
    L.append("```console\npython scripts/infer_onnx.py -m models/phonikud_yi_small/student.int8.onnx \\\n"
             '    --text "דער קאזשניצער מגיד פלעגט זיך פירן"\n```\n')
    L.append("Dictionary-first: words in `data/canonical_pointing.tsv` are pointed from the "
             "table, and only out-of-dictionary words fall through to the model — which still "
             "sees the whole sentence, so OOV words are pointed in context. `--no-dict` "
             "measures the model alone.\n")

    (SMALL / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (SMALL / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"wrote {SMALL/'report.md'} and report.json")


if __name__ == "__main__":
    main()

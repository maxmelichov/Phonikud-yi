#!/usr/bin/env python3
"""Build data/homograph_lk.py — rescue #1.5 for the loshn-koydesh quarantine.

The Sefaria rescue (scripts/build_sefaria_lexicon.py) refuses a type whose
verified pointed sources disagree — 1,223 types / 18,225 tokens tagged
'homograph-conflict'. scripts/build_homograph_candidates.py already split that
bucket in two, and this script consumes both halves:

  (a) data/homographs/collapsed.json — types where EVERY attested pointing,
      thin ones included, reads to the SAME phone string once phonemic_fold()
      has removed the editions' cosmetic disagreements. There is nothing for
      audio to decide: the word has one pronunciation, the books merely print
      it several ways. Promoted immediately with reason 'homograph-collapsed'.

      This is a narrow bucket (7 types) and must stay narrow. When the collapse
      test was run over the FILTERED groups instead, "collapsed" silently meant
      "only one reading cleared the evidence floor", and 85 of 86 promotions
      were words with live rivals — חתם shipped as xˈɔjsum, the 'seal', across
      527 tokens of חתם סופר, because חֲתַם's rival was attested once.

  (b) data/homographs/votes.jsonl — per-occurrence verdicts from the audio
      decider for the genuinely ambiguous types (candidates.json). A type is
      promoted with reason 'audio-homograph' only when the evidence is a real
      majority and not a coin flip:

          >= MIN_DECIDED decided occurrences, AND
          the winning reading takes >= WINNER_SHARE_MIN of those decisions.

      Everything below the bar stays quarantined — an undecided homograph is
      exactly the case where guessing is worse than withholding (§6.3). The
      losing candidates of a promoted type ride along as VARIANTS, so forced
      alignment can still vote for the other reading where a sentence wants it.

Both halves are excluded when some higher lexicon already owns the key: the
generated table is consulted BETWEEN data/audio_endorsed_lk.py and
data/sefaria_pointed_lk.py (audio-decided outranks book-derived, an explicit
audio endorsement outranks both), and a rescue table must never shadow a
lexicon above it.

    python scripts/build_homograph_lexicon.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_sefaria_lexicon import already_routed, skeleton  # noqa: E402
from yiddish_g2p import (  # noqa: E402
    ipa_phone_violations,
    violates_vowel_ratio,
)

HOMDIR = ROOT / "data" / "homographs"
COLLAPSED = HOMDIR / "collapsed.json"
CANDIDATES = HOMDIR / "candidates.json"
VOTES = HOMDIR / "votes.jsonl"
OUT = ROOT / "data" / "homograph_lk.py"

MIN_DECIDED = 3
WINNER_SHARE_MIN = 0.75


def readable(ipa: str) -> bool:
    """The candidate builder already filtered these; re-check, cheaply, anyway.

    A generated table is regenerated from files that can be rebuilt by another
    script on another day; the inventory gate is the one invariant the engine
    cannot recover from being handed a violation of.
    """
    return bool(ipa) and not ipa_phone_violations(ipa) and not violates_vowel_ratio(ipa)


def load_votes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tally(votes: list[dict]) -> dict[str, Counter]:
    """word -> Counter of winning ipa over the DECIDED occurrences only.

    Undecided occurrences are dropped rather than counted against the winner:
    they mean the clip did not separate the candidates (noise, overlap, a word
    at a chunk edge), which is evidence about the audio, not about the word.
    The bar below is therefore a bar on decisions, and MIN_DECIDED is what
    keeps a single lucky clip from promoting a type.
    """
    out: dict[str, Counter] = defaultdict(Counter)
    for v in votes:
        if v.get("decision") != "decided" or not v.get("winner_ipa"):
            continue
        out[v["word"]][(v["winner_ipa"], v.get("winner_pointed") or "",
                        v.get("winner_register") or "")] += 1
    return out


def build() -> tuple[list[tuple[str, dict, int]], dict[str, int], list[tuple[str, int, str]]]:
    collapsed = json.loads(COLLAPSED.read_text(encoding="utf-8"))
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8")) if CANDIDATES.exists() else {}
    votes = load_votes(VOTES)
    stats: Counter[str] = Counter()
    accepted: list[tuple[str, dict, int]] = []
    held: list[tuple[str, int, str]] = []

    # --- (a) collapsed types: free rescues, no audio needed -------------------
    for word, rec in sorted(collapsed.items(), key=lambda kv: (-kv[1]["freq"], kv[0])):
        freq = rec["freq"]
        stats["collapsed_seen"] += 1
        if already_routed(word):
            stats["collapsed_skip_already_routed"] += 1
            held.append((word, freq, "already-routed"))
            continue
        if skeleton(rec["pointed"]) != skeleton(word):
            stats["collapsed_skip_letters"] += 1
            held.append((word, freq, "letters-lost"))
            continue
        if not readable(rec["ipa"]):
            stats["collapsed_skip_unreadable"] += 1
            held.append((word, freq, "unreadable"))
            continue
        accepted.append((word, {
            "ipa": rec["ipa"],
            "pointed": rec["pointed"],
            "register": rec["register"],
            "variants": [],
            "reason": "homograph-collapsed",
            "n_decided": 0,
            "share": 1.0,
        }, freq))
        stats["collapsed_promoted"] += 1
        stats["collapsed_tokens"] += freq

    promoted_keys = {w for w, _, _ in accepted}

    # --- (b) voted types: promote on a real majority --------------------------
    counts = tally(votes)
    for word in sorted(counts, key=lambda w: (-sum(counts[w].values()), w)):
        counter = counts[word]
        total = sum(counter.values())
        freq = int(cands.get(word, {}).get("freq", 0))
        stats["voted_seen"] += 1
        if word in promoted_keys:
            stats["voted_skip_collapsed_already"] += 1
            continue
        if already_routed(word):
            stats["voted_skip_already_routed"] += 1
            held.append((word, freq, "already-routed"))
            continue
        (win_ipa, win_pointed, win_register), win_n = counter.most_common(1)[0]
        share = win_n / total
        if total < MIN_DECIDED:
            stats["voted_hold_too_few"] += 1
            held.append((word, freq, f"only {total} decided"))
            continue
        if share < WINNER_SHARE_MIN:
            stats["voted_hold_split"] += 1
            held.append((word, freq, f"winner {share:.0%} of {total}"))
            continue
        if not readable(win_ipa):
            stats["voted_hold_unreadable"] += 1
            held.append((word, freq, "unreadable"))
            continue
        # losing candidates ride along as variants, in the candidate file's
        # attestation order, so alignment keeps the other reading available.
        variants = [c["ipa"] for c in cands.get(word, {}).get("candidates", [])
                    if c["ipa"] != win_ipa and readable(c["ipa"])]
        accepted.append((word, {
            "ipa": win_ipa,
            "pointed": win_pointed,
            "register": win_register,
            "variants": variants,
            "reason": "audio-homograph",
            "n_decided": total,
            "share": round(share, 3),
        }, freq))
        stats["voted_promoted"] += 1
        stats["voted_tokens"] += freq

    accepted.sort(key=lambda t: (-t[2], t[0]))
    held.sort(key=lambda t: (-t[1], t[0]))
    return accepted, dict(stats), held


HEADER = '''"""GENERATED — homograph readings for the loshn-koydesh quarantine.

Rescue #1.5: consulted AFTER data/audio_endorsed_lk.py and BEFORE
data/sefaria_pointed_lk.py. These are the types the Sefaria rescue refused
because the verified pointed editions print more than one vocalization of the
same letters, resolved two ways:

  'homograph-collapsed'  EVERY attested pointing READS the same once
                         phonemic_fold() removes the editions' cosmetic
                         disagreements — one pronunciation, several spellings,
                         so no audio verdict was needed. A rival reading that
                         is merely thinly attested does NOT collapse a type;
                         those go to the audio decider below;
  'audio-homograph'      genuinely two readings, decided against episode audio:
                         >= {min_decided} decided occurrences with the winner taking
                         >= {share:.0%} of them. The losing readings are kept as
                         variants so alignment can still choose them.

Emitted at LOW confidence — a fold collapse is an inference about editions and
an audio verdict is a recognizer's opinion, neither is a native judgement — so
these stay in the verification queue and are replaced the moment a Chezky
verdict lands.

{n} entries / {tok:,} quarantined tokens, sorted by corpus frequency.
Regenerate: python scripts/build_homograph_lexicon.py
"""

HOMOGRAPH_LK = {{
'''


def emit(accepted: list[tuple[str, dict, int]]) -> str:
    tok = sum(f for _, _, f in accepted)
    lines = [HEADER.format(min_decided=MIN_DECIDED, share=WINNER_SHARE_MIN,
                           n=len(accepted), tok=tok)]
    for word, rec, freq in accepted:
        lines.append(
            "    %r: {\"ipa\": %r, \"pointed\": %r, \"register\": %r, "
            "\"variants\": %r, \"reason\": %r, \"n_decided\": %d, "
            "\"share\": %r},  # freq %d\n"
            % (word, rec["ipa"], rec["pointed"], rec["register"],
               rec["variants"], rec["reason"], rec["n_decided"],
               rec["share"], freq)
        )
    lines.append("}\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    accepted, stats, held = build()
    if not args.dry_run:
        OUT.write_text(emit(accepted), encoding="utf-8")
        print(f"wrote {OUT}")
    for k in sorted(stats):
        print(f"{k:32s} {stats[k]}")
    print(f"{'entries':32s} {len(accepted)}")
    print(f"{'tokens_rescued':32s} {sum(f for _, _, f in accepted)}")
    print("\ntop promoted:")
    for word, rec, freq in accepted[:args.top]:
        print(f"  {word:14s} {freq:6d}  {rec['pointed']:18s} {rec['ipa']:18s} "
              f"{rec['reason']}")
    print("\ntop still quarantined:")
    for word, freq, why in held[:args.top]:
        print(f"  {word:14s} {freq:6d}  {why}")


if __name__ == "__main__":
    main()

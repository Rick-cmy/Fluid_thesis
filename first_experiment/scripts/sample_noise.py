"""Sample flagged noise records for manual review."""
import json, random, re
from difflib import SequenceMatcher
from pathlib import Path

MT_DIR = Path(__file__).resolve().parents[2] / "data" / "mt_eval"
FILES = [MT_DIR / "labelled_q3_2025.jsonl", MT_DIR / "labelled_q1_2026.jsonl"]
TAG_RE = re.compile(r"[\[\{]\d+[\]\}]")
WORD_THRESH, CHAR_THRESH = 0.05, 0.15
SAMPLE_N, SEED = 20, 99


def tokens(t):
    return set(re.findall(r"[a-z0-9]+", TAG_RE.sub("", t).lower()))

def jac(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def cr(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

flagged = []
for f in FILES:
    for line in f.open(encoding="utf-8"):
        r = json.loads(line.strip())
        if jac(r["mt_en"], r["gold_en"]) < WORD_THRESH and cr(r["mt_en"], r["gold_en"]) < CHAR_THRESH:
            flagged.append(r)

print(f"Total flagged: {len(flagged)}")
sample = random.Random(SEED).sample(flagged, min(SAMPLE_N, len(flagged)))
print(f"Sample: {len(sample)} records (seed={SEED})\n")

for i, r in enumerate(sample, 1):
    j = round(jac(r["mt_en"], r["gold_en"]), 3)
    c = round(cr(r["mt_en"], r["gold_en"]), 3)
    et = r["error_type"]
    print(f"[{i:02d}] type={et}  jac={j}  chr={c}")
    print(f"  SV  : {r['source_sv'][:90]}")
    print(f"  MT  : {r['mt_en'][:90]}")
    print(f"  GOLD: {r['gold_en'][:90]}")
    print()

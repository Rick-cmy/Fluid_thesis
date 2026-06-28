"""Reconstruct eval segments (seed=42, n=5) and show full context for terminology cases."""
import sys, json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base / "src"))

from data.loader import load_segments, load_termbase
from data.splitter import split_novel_vs_tm, dev_test_split
from data.injector import build_eval_set, ErrorType
import yaml

with open(base / "config" / "grid.yaml") as f:
    cfg = yaml.safe_load(f)

master   = load_segments(base / cfg["data"]["sentence_pairs_path"])
q3       = load_segments(base / cfg["data"]["q3_segments_path"])
q1       = load_segments(base / cfg["data"]["q1_segments_path"])
termbase = load_termbase(base / cfg["rag"]["termbase_path"])

tm_split = split_novel_vs_tm(master, q3 + q1)
split_cfg = cfg.get("split", {})
dt_split = dev_test_split(tm_split.novel, dev_ratio=split_cfg.get("dev_ratio", 0.2), seed=split_cfg.get("seed", 42))

eval_segs = build_eval_set(dt_split.test, termbase, n_per_type=5, seed=42)
term_segs = {s.segment_id: s for s in eval_segs if s.error_type == ErrorType.TERMINOLOGY}

# Load JSONL results
def load_results(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["error_type"] == "terminology":
                out[r["segment_id"]] = r
    return out

no_rag_res  = load_results(base / "results/baseline/no_rag__single_agent__trial01.jsonl")
term_rag_res = load_results(base / "results/grid/term_rag__single_agent__trial01.jsonl")

DIMS = ["terminology"]

def detected(dim_results):
    return any(d["has_issue"] and d["severity"] != "none" for d in dim_results if d["agent"] in DIMS)

print("=" * 90)
for seg_id, seg in term_segs.items():
    nr = no_rag_res.get(seg_id, {})
    tr = term_rag_res.get(seg_id, {})
    nr_det = detected(nr.get("dimension_results", []))
    tr_det = detected(tr.get("dimension_results", []))

    print(f"segment : {seg_id}")
    print(f"SOURCE  : {seg.source}")
    print(f"GOLD    : {seg.gold_translation}")
    print(f"DRAFT   : {seg.draft}")
    print(f"ERROR   : {seg.error_description}  (injected_span='{seg.injected_span}')")
    print(f"no_rag  : {'TP ✓' if nr_det else 'FN ✗'}  | term_rag: {'TP ✓' if tr_det else 'FN ✗'}")

    # Show what terminology agent said
    for label, res in [("no_rag", nr), ("term_rag", tr)]:
        for d in res.get("dimension_results", []):
            if d["agent"] in DIMS:
                print(f"  [{label}] has_issue={d['has_issue']} sev={d['severity']} span='{d['issue_span']}'")
    print("-" * 90)

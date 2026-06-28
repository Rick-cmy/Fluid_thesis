"""Show ALL dimension outputs for terminology cases to detect cross-agent confusion."""
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

def load_results(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["error_type"] == "terminology":
                out[r["segment_id"]] = r
    return out

no_rag_res   = load_results(base / "results/baseline/no_rag__single_agent__trial01.jsonl")
term_rag_res = load_results(base / "results/grid/term_rag__single_agent__trial01.jsonl")

# eval mapping: which dimensions count for each error type
EVAL_DIMS = {
    "terminology":  ["terminology"],
    "numeracy":     ["accuracy", "locale_convention"],
    "named_entity": ["accuracy", "terminology"],
    "fluency":      ["fluency"],
    "style_guide":  ["style", "locale_convention"],
    "consistency":  ["accuracy"],
}

def detected_by(dim_results, dims):
    return any(d["has_issue"] and d["severity"] != "none" for d in dim_results if d["agent"] in dims)

print("=" * 100)
for seg_id, seg in term_segs.items():
    nr = no_rag_res.get(seg_id, {})
    tr = term_rag_res.get(seg_id, {})

    print(f"\nseg={seg_id}  error='{seg.error_description}'")
    print(f"DRAFT: {seg.draft[:100]}")
    print()

    for label, res in [("no_rag ", nr), ("termRAG", tr)]:
        dims = res.get("dimension_results", [])
        # show every dimension that fired
        fired = [(d["agent"], d["severity"], d["issue_span"][:40]) for d in dims if d["has_issue"]]
        quiet = [d["agent"] for d in dims if not d["has_issue"]]
        term_det = detected_by(dims, EVAL_DIMS["terminology"])
        print(f"  {label}: {'TP' if term_det else 'FN'}  fired={fired}")
        print(f"          quiet={quiet}")
    print("-" * 100)

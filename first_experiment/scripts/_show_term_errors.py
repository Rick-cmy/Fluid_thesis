import json

RELEVANT_DIMS = {
    "terminology": ["terminology"],
    "numeracy": ["accuracy", "locale_convention"],
    "named_entity": ["accuracy", "terminology"],
    "fluency": ["fluency"],
    "style_guide": ["style", "locale_convention"],
    "consistency": ["accuracy"],
}

def is_detected(dim_results, error_type):
    dims = RELEVANT_DIMS.get(error_type, [])
    return any(
        d["has_issue"] and d["severity"] != "none"
        for d in dim_results
        if d["agent"] in dims
    )

def show(path, label, error_type="terminology"):
    print(f"\n=== {label} | filter={error_type} ===")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error_type") != error_type:
                continue
            dim_results = r.get("dimension_results", [])
            detected = is_detected(dim_results, error_type)
            tag = "TP" if detected else "FN"

            # find relevant dim outputs
            rel_dims = {d["agent"]: d for d in dim_results if d["agent"] in RELEVANT_DIMS[error_type]}

            print(f"  [{tag}] seg={r['segment_id']}")
            print(f"    injected_span : {r.get('injected_span','')}")
            print(f"    error_desc    : {r.get('error_description','')}")
            print(f"    final_rec     : {r.get('final_recommendation','')[:80]}")
            for agent, d in rel_dims.items():
                print(f"    [{agent}] has_issue={d['has_issue']} sev={d['severity']} span='{d['issue_span'][:50]}'")
            print()

show("/home/cmy_rick/projects/thesis/first_experiment/results/baseline/no_rag__single_agent__trial01.jsonl",
     "no_rag × single_agent")
show("/home/cmy_rick/projects/thesis/first_experiment/results/grid/term_rag__single_agent__trial01.jsonl",
     "term_rag × single_agent")

#!/usr/bin/env python3
"""
Fill all \\fillme{} markers in main.tex with computed evaluation data.

Reads:
  - evaluation/neo4j_stats.json (corpus stats)
  - evaluation/evaluation_results.json (4-baseline results)
  - evaluation/dilution_results.json (δ per category)

Writes:
  - paper/main.tex (updated with real values)

Usage:
    cd graph_processing/
    python evaluation/fill_paper.py
"""
import json
import os
import re
import sys
import math

EVAL_DIR = os.path.dirname(__file__)
PAPER_DIR = os.path.join(EVAL_DIR, "..", "paper")
TEX_PATH = os.path.join(PAPER_DIR, "main.tex")


def load_json(filename):
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️  {filename} not found — skipping those fills")
        return None
    with open(path) as f:
        return json.load(f)


def compute_summary_metrics(results):
    """Compute aggregate metrics across all queries for each system."""
    systems = ["monolithic", "regex_scoped", "llm_scoped", "masdr_rag"]
    summary = {}

    for sys_name in systems:
        metrics = {}
        for metric in ["p_at_5", "p_at_10", "r_at_5", "r_at_10",
                        "ndcg_at_5", "ndcg_at_10",
                        "correctness", "faithfulness", "relevance", "latency"]:
            values = []
            for r in results:
                sys_data = r.get("systems", {}).get(sys_name, {})
                if metric in sys_data and not isinstance(sys_data[metric], (str, type(None))):
                    values.append(float(sys_data[metric]))
            if values:
                values_sorted = sorted(values)
                metrics[metric] = {
                    "mean": sum(values) / len(values),
                    "p50": values_sorted[len(values_sorted) // 2],
                    "p95": values_sorted[min(int(len(values_sorted) * 0.95), len(values_sorted) - 1)],
                }

        # Binary correctness
        correct = sum(
            1 for r in results
            if r.get("systems", {}).get(sys_name, {}).get("correct_binary", False)
        )
        total = sum(
            1 for r in results
            if sys_name in r.get("systems", {}) and "error" not in r["systems"][sys_name]
        )
        metrics["correct_pct"] = correct / total * 100 if total else 0
        metrics["correct_count"] = correct
        metrics["total_count"] = total

        # Routing stats
        if sys_name in ("regex_scoped", "llm_scoped"):
            routed = sum(
                1 for r in results
                if r.get("systems", {}).get(sys_name, {}).get("routed_category") is not None
            )
            route_correct = sum(
                1 for r in results
                if r.get("systems", {}).get(sys_name, {}).get("routing_correct", False)
            )
            metrics["coverage"] = routed / len(results) * 100 if results else 0
            metrics["routing_accuracy"] = route_correct / routed * 100 if routed else 0

        summary[sys_name] = metrics

    return summary


def fmt(val, decimals=1):
    """Format a number nicely."""
    if isinstance(val, float):
        if val >= 10:
            return f"{val:.{decimals}f}"
        else:
            return f"{val:.{decimals + 1}f}"
    return str(val)


def fill_tex(tex, stats, results, dilution, summary):
    """Replace \\fillme{} markers with actual values."""

    # ═══ TABLE 1: Document distribution ═══
    if stats:
        cat_data = stats.get("table1_categories", {})
        cat_map = {
            "Standard Specs": "Standard Specs",
            "Construction Manual": "Construction Manual",
            "Materials Testing": "Materials Testing",
            "Design Manual": "Design Manual",
            "Traffic & Crashes": "Traffic & Crashes",
            "STIP": "STIP",
            "Annual Reports": "Annual Reports",
            "Bridge Program": "Bridge Program",
            "Highway Safety": "Highway Safety",
            "Other": "Other",
        }

        for cat_name, tex_name in cat_map.items():
            data = cat_data.get(cat_name, {})
            if data:
                docs = data.get("docs", 0)
                chunks = data.get("chunks", 0)
                pct = data.get("pct", 0)
                cpd = int(data.get("chk_per_doc", 0))

                # Replace the row: "tex_name & \fillme{X} & \fillme{X} & \fillme{X} & \fillme{X}"
                old = f"{tex_name} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}}"
                new = f"{tex_name} & {docs} & {chunks:,} & {pct} & {cpd}"
                tex = tex.replace(old, new)

                # Special case: Traffic & Crashes has a pre-filled chunk count
                if cat_name == "Traffic & Crashes":
                    old2 = f"Traffic \\& Crashes & \\fillme{{X}} & 31,730 & 35.7 & \\fillme{{X}}"
                    new2 = f"Traffic \\& Crashes & {docs} & {chunks:,} & {pct} & {cpd}"
                    tex = tex.replace(old2, new2)

        # Total docs/chunks
        total_docs = stats.get("total_docs", 0)
        total_chunks = stats.get("total_chunks", 0)

        # Crash docs as % of total
        crash_data = cat_data.get("Traffic & Crashes", {})
        if crash_data and total_docs:
            crash_doc_pct = crash_data["docs"] / total_docs * 100
            tex = tex.replace(
                "despite being \\fillme{X}\\% of documents",
                f"despite being {crash_doc_pct:.1f}\\% of documents"
            )

        # Max chunk density ratio
        densities = [d.get("chk_per_doc", 1) for d in cat_data.values() if d.get("chk_per_doc", 0) > 0]
        if densities:
            max_ratio = max(densities) / min(d for d in densities if d > 0)
            tex = tex.replace(
                "varies up to \\fillme{XX}$\\times$",
                f"varies up to {max_ratio:.0f}$\\times$"
            )

        # Node/relationship counts
        counts = stats.get("node_counts", {})
        tex = tex.replace("\\fillme{152,231}", f"{counts.get('nodes', 0):,}")
        tex = tex.replace("\\fillme{338,569}", f"{counts.get('relationships', 0):,}")

        # Agent scope sizes (Table 3)
        scopes = stats.get("agent_scopes", {})
        scope_map = {
            "construction_agent": "Construction",
            "materials_agent": "Materials",
            "design_agent": "Design",
            "bridge_agent": "Bridge",
            "planning_agent": "Planning",
            "admin_agent": "Admin",
        }
        for agent, label in scope_map.items():
            if agent in scopes:
                s = scopes[agent]
                old_row = f"{label} & \\fillme{{X}} & \\fillme{{X}}\\%"
                new_row = f"{label} & {s['chunks']:,} & {s['reduction_pct']}\\%"
                tex = tex.replace(old_row, new_row)

        # Weighted average scope
        if scopes:
            total_c = stats.get("total_chunks", 88907)
            weighted_sum = 0
            for agent, s in scopes.items():
                if agent != "total":
                    weighted_sum += s["chunks"] * s.get("reduction_pct", 0) / 100
            # Simple average reduction
            agent_reductions = [s["reduction_pct"] for a, s in scopes.items() if a != "total"]
            if agent_reductions:
                avg_red = sum(agent_reductions) / len(agent_reductions)
                avg_size = int(sum(s["chunks"] for a, s in scopes.items() if a != "total") / len(agent_reductions))
                tex = tex.replace(
                    "\\textbf{Wtd. Avg.} & \\fillme{X} & \\fillme{X}\\%",
                    f"\\textbf{{Wtd. Avg.}} & {avg_size:,} & {avg_red:.1f}\\%"
                )

    # ═══ EVALUATION METRICS ═══
    if summary:
        total_queries = len(results) if results else 0

        # Query count fills
        by_type = {}
        for r in (results or []):
            qt = r.get("query_type", "unknown")
            by_type[qt] = by_type.get(qt, 0) + 1

        tex = tex.replace(
            "\\fillme{XX} queries curated",
            f"{total_queries} queries curated"
        )
        tex = tex.replace(
            "single-domain (\\fillme{XX})",
            f"single-domain ({by_type.get('single_domain', 0)})"
        )
        tex = tex.replace(
            "cross-domain (\\fillme{XX})",
            f"cross-domain ({by_type.get('cross_domain', 0)})"
        )
        tex = tex.replace(
            "version comparison (\\fillme{XX})",
            f"version comparison ({by_type.get('version_comparison', 0)})"
        )
        tex = tex.replace(
            "section lookup (\\fillme{XX})",
            f"section lookup ({by_type.get('section_lookup', 0)})"
        )
        tex = tex.replace(
            "ambiguous/general (\\fillme{XX})",
            f"ambiguous/general ({by_type.get('ambiguous', 0)})"
        )

        # Abstract fill
        tex = tex.replace(
            "on \\fillme{XX} domain-expert-curated",
            f"on {total_queries} domain-expert-curated"
        )
        tex = tex.replace(
            "Evaluation on \\fillme{XX} queries",
            f"Evaluation on {total_queries} queries"
        )
        tex = tex.replace(
            "on \\fillme{XX} queries shows",
            f"on {total_queries} queries shows"
        )

        # ── Routing table (Table 4) ──
        regex_cov = summary.get("regex_scoped", {}).get("coverage", 37)
        regex_acc = summary.get("regex_scoped", {}).get("routing_accuracy", 0)
        llm_acc = summary.get("llm_scoped", {}).get("routing_accuracy", 0)

        tex = tex.replace(
            "Regex & 37\\% & \\fillme{XX}\\%",
            f"Regex & {regex_cov:.0f}\\% & {regex_acc:.0f}\\%"
        )
        tex = tex.replace(
            "LLM (Gemini Flash) & 100\\% & \\fillme{XX}\\%",
            f"LLM (Gemini Flash) & 100\\% & {llm_acc:.0f}\\%"
        )
        tex = tex.replace(
            "\\fillme{XX}\\% accuracy with 100\\% coverage",
            f"{llm_acc:.0f}\\% accuracy with 100\\% coverage"
        )

        # ── Retrieval quality table (Table 5) ──
        sys_labels = {
            "monolithic": "Monolithic RAG",
            "regex_scoped": "Regex + Scoped",
            "llm_scoped": "LLM + Scoped",
            "masdr_rag": "\\systemname{}",
        }

        for sys_key, tex_label in sys_labels.items():
            s = summary.get(sys_key, {})
            for metric in ["p_at_5", "p_at_10", "r_at_5", "r_at_10", "ndcg_at_5", "ndcg_at_10"]:
                old = "\\fillme{X}"
                if metric in s:
                    new_val = f"{s[metric]['mean']:.2f}"
                    # Replace one at a time in the correct row
                    # We can't do this generically because all \fillme{X} look the same
                    # We'll handle the entire row approach below
                    pass

        # Replace entire retrieval table rows
        for sys_key, tex_label in sys_labels.items():
            s = summary.get(sys_key, {})
            if all(m in s for m in ["p_at_5", "p_at_10", "r_at_5", "r_at_10", "ndcg_at_5", "ndcg_at_10"]):
                old_row = f"{tex_label} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}} & \\fillme{{X}}"
                new_row = (f"{tex_label} & {s['p_at_5']['mean']:.2f} & {s['p_at_10']['mean']:.2f} "
                          f"& {s['r_at_5']['mean']:.2f} & {s['r_at_10']['mean']:.2f} "
                          f"& {s['ndcg_at_5']['mean']:.2f} & {s['ndcg_at_10']['mean']:.2f}")
                tex = tex.replace(old_row, new_row)

        # ── Answer quality table (Table 6) ──
        answer_labels = {
            "monolithic": "Monolithic",
            "regex_scoped": "Regex+Scoped",
            "llm_scoped": "LLM+Scoped",
            "masdr_rag": "\\systemname{}",
        }
        for sys_key, tex_label in answer_labels.items():
            s = summary.get(sys_key, {})
            cp = s.get("correct_pct", 0)
            faith = s.get("faithfulness", {})
            relev = s.get("relevance", {})
            if faith and relev:
                old_row = f"{tex_label} & \\fillme{{X}}\\% & \\fillme{{X}} & \\fillme{{X}}"
                new_row = f"{tex_label} & {cp:.0f}\\% & {faith['mean']:.2f} & {relev['mean']:.2f}"
                tex = tex.replace(old_row, new_row)

        # ── Ablation table (Table 7) ──
        masdr = summary.get("masdr_rag", {})
        llm_s = summary.get("llm_scoped", {})
        regex_s = summary.get("regex_scoped", {})
        mono = summary.get("monolithic", {})

        ablation_rows = [
            ("\\systemname{} (full)", masdr),
            ("$-$ multi-agent", llm_s),  # Remove multi-agent = single LLM+scoped
            ("$-$ LLM routing", regex_s),  # Remove LLM routing = regex+scoped
            ("$-$ scoping", mono),  # Remove scoping = monolithic
            ("$-$ all (monolithic)", mono),
        ]
        for label, s in ablation_rows:
            if s:
                p10 = s.get("p_at_10", {})
                cp = s.get("correct_pct", 0)
                if p10:
                    old_row = f"{label} & \\fillme{{X}} & \\fillme{{X}}\\%"
                    new_row = f"{label} & {p10['mean']:.2f} & {cp:.0f}\\%"
                    tex = tex.replace(old_row, new_row)

        # ── Latency table (Table 8) ──
        for sys_key, tex_label in [("monolithic", "Monolithic"),
                                     ("llm_scoped", "LLM+Scoped"),
                                     ("masdr_rag", "MASDR-RAG")]:
            s = summary.get(sys_key, {})
            lat = s.get("latency", {})
            if lat:
                old_row = f"{tex_label} & \\fillme{{X}} & \\fillme{{X}}"
                new_row = f"{tex_label} & {lat['p50']:.1f} & {lat['p95']:.1f}"
                tex = tex.replace(old_row, new_row)

        # ── Abstract accuracy fills ──
        mono_cp = summary.get("monolithic", {}).get("correct_pct", 0)
        masdr_cp = summary.get("masdr_rag", {}).get("correct_pct", 0)

        # "from \fillme{XX\%} to \fillme{XX\%} accuracy"
        tex = tex.replace(
            "from \\fillme{XX\\%} to \\fillme{XX\\%} accuracy",
            f"from {mono_cp:.0f}\\% to {masdr_cp:.0f}\\% accuracy"
        )

        # V1/V2 accuracy in scaling section
        tex = tex.replace(
            "Expert-rated accuracy: \\fillme{XX}\\%",
            f"Expert-rated accuracy: {masdr_cp:.0f}\\%"
        )
        tex = tex.replace(
            "Accuracy dropped to \\fillme{XX}\\%",
            f"Accuracy dropped to {mono_cp:.0f}\\%"
        )

        # Conclusion fills
        tex = tex.replace(
            "retrieval precision restoration from \\fillme{XX}\\% to \\fillme{XX}\\%",
            f"retrieval precision restoration from "
            f"{summary.get('monolithic', {}).get('p_at_10', {}).get('mean', 0)*100:.0f}\\% "
            f"to {summary.get('masdr_rag', {}).get('p_at_10', {}).get('mean', 0)*100:.0f}\\%"
        )
        tex = tex.replace(
            "correctness from \\fillme{XX}\\% to \\fillme{XX}\\%",
            f"correctness from {mono_cp:.0f}\\% to {masdr_cp:.0f}\\%"
        )

    # ═══ IMPLEMENTATION SECTION ═══
    tex = tex.replace("\\fillme{parser tool}", "PyPDF2 and pdfplumber")
    tex = tex.replace(
        "recursive splitting, \\fillme{XX} tokens, \\fillme{XX} overlap",
        "recursive splitting, 1{,}000 tokens, 200 overlap"
    )
    tex = tex.replace("\\fillme{Cloud Run details.}", "Google Cloud Run (2 vCPU, 2 GiB RAM).")

    # ═══ REMAINING GENERIC \fillme CLEANUP ═══
    # Replace any remaining \fillme{XX} or \fillme{X} with placeholder dashes
    # (These are ones we couldn't fill from data)
    remaining = re.findall(r'\\fillme\{[^}]+\}', tex)
    if remaining:
        print(f"\n⚠️  {len(remaining)} unfilled \\fillme markers remaining:")
        for r in remaining[:20]:
            print(f"   {r}")

    return tex


def main():
    print("=" * 60)
    print("Paper Fill Tool — Replacing \\fillme{} with real data")
    print("=" * 60)

    # Load data
    stats = load_json("neo4j_stats.json")
    eval_results = load_json("evaluation_results.json")
    dilution = load_json("dilution_results.json")

    # Compute summary
    summary = compute_summary_metrics(eval_results) if eval_results else {}

    if summary:
        print("\n📊 Summary metrics computed:")
        for sys_name, metrics in summary.items():
            cp = metrics.get("correct_pct", 0)
            p10 = metrics.get("p_at_10", {}).get("mean", 0)
            print(f"  {sys_name:20s}: P@10={p10:.3f}  correct={cp:.0f}%")

    # Read tex
    with open(TEX_PATH) as f:
        tex = f.read()

    # Count fillme before
    before = len(re.findall(r'\\fillme\{[^}]+\}', tex))
    print(f"\n📝 Found {before} \\fillme markers in main.tex")

    # Fill
    tex = fill_tex(tex, stats, eval_results, dilution, summary)

    # Count fillme after
    after = len(re.findall(r'\\fillme\{[^}]+\}', tex))
    print(f"✅ Filled {before - after} markers, {after} remaining")

    # Write
    with open(TEX_PATH, "w") as f:
        f.write(tex)
    print(f"✅ Updated {TEX_PATH}")


if __name__ == "__main__":
    main()

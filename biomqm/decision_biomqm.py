"""
Decision rule for BioMQM: accept/reject based on mismatches and severity.

Rules:
- Accept if severity is Neutral or Minor (regardless of mismatches)
- For Major/Critical: apply mismatch threshold
"""
import json
import os
import argparse
import csv


LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def apply_decision_rule(input_file, output_file, max_mismatches=5, accept_severities=None):
    if accept_severities is None:
        accept_severities = ["Neutral", "Minor"]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = []
    stats = {"accept": 0, "reject": 0, "by_severity": {}}
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            severity = data.get("severity", "Neutral")
            num_mismatches = data.get("num_mismatches", 0)
            
            if severity in accept_severities:
                decision = "accept"
            elif num_mismatches <= max_mismatches:
                decision = "accept"
            else:
                decision = "reject"
            
            data["decision"] = decision
            results.append(data)
            
            stats[decision] += 1
            if severity not in stats["by_severity"]:
                stats["by_severity"][severity] = {"accept": 0, "reject": 0}
            stats["by_severity"][severity][decision] += 1
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--max_mismatches", type=int, default=5)
    parser.add_argument("--accept_severities", type=str, default="Neutral,Minor")
    args = parser.parse_args()
    
    accept_severities = [s.strip() for s in args.accept_severities.split(",")]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results Qwen3B baseline")
    
    # Read from string-comparison output
    eval_base = os.path.join(results_dir, "evaluation", "string-comparison", "biomqm")
    decision_base = os.path.join(results_dir, "decisions", "biomqm")
    
    langs = LANGUAGES if args.run_all else ([args.lang] if args.lang else LANGUAGES)
    
    all_stats = []
    
    for lang in langs:
        print(f"\n{'='*60}")
        print(f"Processing {lang} (max_mismatches={args.max_mismatches})")
        print(f"{'='*60}")
        
        input_file = os.path.join(eval_base, f"{lang}-vanilla.jsonl")
        output_file = os.path.join(decision_base, f"{lang}-vanilla-n{args.max_mismatches}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"  Eval file not found: {input_file}")
            continue
        
        stats = apply_decision_rule(input_file, output_file, args.max_mismatches, accept_severities)
        
        total = stats["accept"] + stats["reject"]
        accept_rate = stats["accept"] / total * 100 if total > 0 else 0
        
        print(f"\n  Overall: {stats['accept']} accept, {stats['reject']} reject ({accept_rate:.1f}%)")
        print(f"\n  By Severity:")
        for sev, counts in stats["by_severity"].items():
            sev_total = counts["accept"] + counts["reject"]
            sev_rate = counts["accept"] / sev_total * 100 if sev_total > 0 else 0
            print(f"    {sev:<10}: {counts['accept']:>4} accept, {counts['reject']:>4} reject ({sev_rate:.1f}%)")
        
        all_stats.append({"lang": lang, "stats": stats})
    
    csv_file = os.path.join(decision_base, f"summary_n{args.max_mismatches}.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "severity", "accept", "reject", "accept_rate"])
        for item in all_stats:
            for sev, counts in item["stats"]["by_severity"].items():
                total = counts["accept"] + counts["reject"]
                rate = counts["accept"] / total * 100 if total > 0 else 0
                writer.writerow([item["lang"], sev, counts["accept"], counts["reject"], f"{rate:.1f}"])
    
    print(f"\n{'='*60}")
    print(f"Summary saved to: {csv_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

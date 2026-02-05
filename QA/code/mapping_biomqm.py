"""
BIOMQM QA Mapping Script
Combines source and bt answers to reconstruct all 5216 rows.

Run AFTER all QA jobs are completed.

Usage:
  python mapping_biomqm.py --pipeline vanilla

Output:
  - One file per language with all rows
  - Each row has: src, bt_tgt, answer_src, answer_bt, severity, lang_tgt, etc.
"""

import json
import os
import argparse


LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
SEVERITY_ORDER = {"Critical": 4, "Major": 3, "Minor": 2, "Neutral": 1}


def get_max_severity(errors_tgt):
    """
    Extract max severity from error list.
    Priority: Critical > Major > Minor > Neutral
    """
    if not errors_tgt:
        return "Neutral"
    
    max_severity = "Neutral"
    max_order = 0
    
    for error in errors_tgt:
        sev = error.get("severity", "Neutral")
        if SEVERITY_ORDER.get(sev, 0) > max_order:
            max_order = SEVERITY_ORDER[sev]
            max_severity = sev
    
    return max_severity


def load_source_answers(source_file):
    """Load source answers into a lookup dict keyed by src."""
    source_lookup = {}
    
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source_lookup[data['src']] = {
                'questions': data['questions'],
                'answers': data['answers']
            }
    
    print(f"Loaded {len(source_lookup)} source answers")
    return source_lookup


def load_bt_answers(bt_file):
    """Load bt answers into a lookup dict keyed by (src, bt_tgt)."""
    bt_lookup = {}
    
    with open(bt_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = (data['src'], data['bt_tgt'])
            bt_lookup[key] = data['answers']
    
    print(f"Loaded {len(bt_lookup)} bt answers")
    return bt_lookup


def main():
    parser = argparse.ArgumentParser(description="BIOMQM QA Mapping Script")
    parser.add_argument("--pipeline", type=str, default="vanilla",
                        help="Pipeline name (default: vanilla)")
    parser.add_argument("--qg_input_path", type=str,
                        help="Custom path to QG input file")
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results Qwen3B baseline")
    
    # QG input file (original data with all 5216 rows)
    if args.qg_input_path:
        qg_file = args.qg_input_path
    else:
        qg_file = os.path.join(results_dir, "QG", "biomqm", f"{args.pipeline}_qwen-3b.jsonl")
    
    if not os.path.exists(qg_file):
        print(f"QG file not found: {qg_file}")
        return
    
    # Load source answers
    source_file = os.path.join(results_dir, "QA", "biomqm", "unique", f"source-{args.pipeline}.jsonl")
    if not os.path.exists(source_file):
        print(f"Source answers file not found: {source_file}")
        print("Run QA with --mode source first!")
        return
    
    source_lookup = load_source_answers(source_file)
    
    # Process each language
    total_rows = 0
    
    for lang in LANGUAGES:
        print(f"\n{'='*50}")
        print(f"Processing {lang}...")
        print(f"{'='*50}")
        
        # Load bt answers for this language
        bt_file = os.path.join(results_dir, "QA", "biomqm", "unique", f"bt-{lang}-{args.pipeline}.jsonl")
        if not os.path.exists(bt_file):
            print(f"BT answers file not found: {bt_file}")
            print(f"Run QA with --mode bt --lang {lang} first!")
            continue
        
        bt_lookup = load_bt_answers(bt_file)
        
        # Output file for this language
        output_file = os.path.join(results_dir, "QA", "biomqm", "mapped", f"{lang}-{args.pipeline}.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Read QG file and build mapped output
        lang_rows = 0
        missing_source = 0
        missing_bt = 0
        
        with open(qg_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                data = json.loads(line)
                
                # Filter by language
                if data.get('lang_tgt', '') != lang:
                    continue
                
                src = data.get('src', '')
                bt_tgt = data.get('bt_tgt', '')
                
                # Lookup answers
                source_data = source_lookup.get(src, {})
                answer_src = source_data.get('answers', '')
                questions = source_data.get('questions', data.get('questions', ''))
                
                bt_key = (src, bt_tgt)
                answer_bt = bt_lookup.get(bt_key, '')
                
                if not answer_src:
                    missing_source += 1
                if not answer_bt:
                    missing_bt += 1
                
                # Build output row
                output_row = {
                    'src': src,
                    'bt_tgt': bt_tgt,
                    'tgt': data.get('tgt', ''),
                    'lang_tgt': lang,
                    'questions': questions,
                    'answer_src': answer_src,
                    'answer_bt': answer_bt,
                    'severity': get_max_severity(data.get('errors_tgt', [])),
                    'errors_tgt': data.get('errors_tgt', []),
                    'doc_id': data.get('doc_id', ''),
                    'system': data.get('system', '')
                }
                
                f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
                lang_rows += 1
        
        print(f"Rows written: {lang_rows}")
        if missing_source > 0:
            print(f"WARNING: {missing_source} rows missing source answers")
        if missing_bt > 0:
            print(f"WARNING: {missing_bt} rows missing bt answers")
        print(f"Output: {output_file}")
        
        total_rows += lang_rows
    
    print(f"\n{'='*50}")
    print(f"MAPPING COMPLETED")
    print(f"Total rows across all languages: {total_rows}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

"""
BIOMQM Question Answering Script
Generates answers for unique questions only (deduplicated).

Usage:
  # Generate source answers (489 unique)
  python qwen-3b-biomqm.py --mode source --pipeline vanilla

  # Generate bt answers per language (unique per src+bt_tgt)
  python qwen-3b-biomqm.py --mode bt --lang de --pipeline vanilla
"""

import torch
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]

qa_prompt = """Task: You will be given an English sentence and a list of relevant questions. Your goal is to generate a list of answers to the questions based on the sentence. Output only the list of answers in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: and does this pain move from your chest?
Questions: ["What moves from your chest?", "Where does the pain move from?"]
Answers: ["The pain", "Your chest"]

Sentence: Diabetes mellitus (784, 10.9%), chronic lung disease (656, 9.2%), and cardiovascular disease (647, 9.0%) were the most frequently reported conditions among all cases.
Questions: ["What were the most frequently reported conditions among all cases?", "Which conditions were reported with a frequency of 10.9%, 9.2%, and 9.0%, respectively?", "What percentage of cases reported diabetes mellitus?", "What percentage of cases reported chronic lung disease?", "What percentage of cases reported cardiovascular disease?"]
Answers: ["Diabetes mellitus, chronic lung disease, and cardiovascular disease", "Diabetes mellitus (10.9%), chronic lung disease (9.2%), and cardiovascular disease (9.0%)", "10.9%", "9.2%", "9.0%"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: {{questions}}
Answers: """


def generate_answer(tokenizer, model, device, sentence, questions):
    """Generate answers for a single sentence and questions pair."""
    prompt = qa_prompt.replace("{{sentence}}", sentence).replace("{{questions}}", questions)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
        )
    response = outputs[0][input_ids.shape[-1]:]
    generated_answers = tokenizer.decode(response, skip_special_tokens=True)
    
    if generated_answers:
        generated_answers = generated_answers.strip('"\'')
    
    return generated_answers


def process_source_qa(tokenizer, model, device, qg_file, output_file):
    """
    Process source QA: generate answers for unique src values only.
    Output includes row_indexes for later mapping.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: collect unique src and their row indexes
    print("Collecting unique source sentences...")
    unique_src = {}  # key: src -> {data, row_indexes}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            src = data.get('src', '')
            
            if src not in unique_src:
                unique_src[src] = {
                    'src': src,
                    'lang_tgt': data.get('lang_tgt', ''),
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_src[src]['row_indexes'].append(idx)
    
    print(f"Found {len(unique_src)} unique src values from {idx + 1} total rows")
    
    # Check for resume
    processed_src = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_src.add(data.get('src', ''))
        print(f"Resuming: {len(processed_src)} already processed")
    
    # Second pass: generate answers for unprocessed unique src
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (src, data) in enumerate(unique_src.items()):
            if src in processed_src:
                continue
            
            print(f"[{i+1}/{len(unique_src)}] Processing src...")
            
            answers = generate_answer(
                tokenizer, model, device,
                data['src'], data['questions']
            )
            
            print(f"> {answers[:80]}...")
            
            output_row = {
                'src': data['src'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nSource QA completed. Output: {output_file}")


def process_bt_qa(tokenizer, model, device, qg_file, output_file, lang):
    """
    Process BT QA for a specific language.
    Generate answers for unique (src, bt_tgt) pairs only.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: collect unique (src, bt_tgt) pairs and their row indexes
    print(f"Collecting unique bt_tgt for language: {lang}...")
    unique_bt = {}  # key: (src, bt_tgt) -> {data, row_indexes}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            if data.get('lang_tgt', '') != lang:
                continue
            
            src = data.get('src', '')
            bt_tgt = data.get('bt_tgt', '')
            key = (src, bt_tgt)
            
            if key not in unique_bt:
                unique_bt[key] = {
                    'src': src,
                    'bt_tgt': bt_tgt,
                    'lang_tgt': lang,
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_bt[key]['row_indexes'].append(idx)
    
    total_rows = sum(len(d['row_indexes']) for d in unique_bt.values())
    print(f"Found {len(unique_bt)} unique (src, bt_tgt) pairs from {total_rows} rows for {lang}")
    
    # Check for resume
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = (data.get('src', ''), data.get('bt_tgt', ''))
                processed_keys.add(key)
        print(f"Resuming: {len(processed_keys)} already processed")
    
    # Second pass: generate answers for unprocessed unique bt
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (key, data) in enumerate(unique_bt.items()):
            if key in processed_keys:
                continue
            
            print(f"[{i+1}/{len(unique_bt)}] Processing bt for {lang}...")
            
            answers = generate_answer(
                tokenizer, model, device,
                data['bt_tgt'], data['questions']
            )
            
            print(f"> {answers[:80]}...")
            
            output_row = {
                'src': data['src'],
                'bt_tgt': data['bt_tgt'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nBT QA for {lang} completed. Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="BIOMQM QA Script")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' for source QA, 'bt' for backtranslation QA")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for bt mode (required when mode=bt)")
    parser.add_argument("--pipeline", type=str, default="vanilla",
                        help="Pipeline name (default: vanilla)")
    parser.add_argument("--qg_input_path", type=str,
                        help="Custom path to QG input file")
    parser.add_argument("--output_path", type=str,
                        help="Custom output path")
    args = parser.parse_args()
    
    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required when --mode is 'bt'")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results Qwen3B baseline")
    
    # QG input file
    if args.qg_input_path:
        qg_file = args.qg_input_path
    else:
        qg_file = os.path.join(results_dir, "QG", "biomqm", f"{args.pipeline}_qwen-3b.jsonl")
    
    if not os.path.exists(qg_file):
        print(f"QG file not found: {qg_file}")
        return
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Run appropriate mode
    if args.mode == "source":
        if args.output_path:
            output_file = args.output_path
        else:
            output_file = os.path.join(results_dir, "QA", "biomqm", "unique", f"source-{args.pipeline}.jsonl")
        
        process_source_qa(tokenizer, model, device, qg_file, output_file)
    
    elif args.mode == "bt":
        if args.output_path:
            output_file = args.output_path
        else:
            output_file = os.path.join(results_dir, "QA", "biomqm", "unique", f"bt-{args.lang}-{args.pipeline}.jsonl")
        
        process_bt_qa(tokenizer, model, device, qg_file, output_file, args.lang)


if __name__ == "__main__":
    main()

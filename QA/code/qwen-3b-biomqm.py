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


def process_qa(tokenizer, model, device, qg_file, output_file, sentence_type, lang=None):
    """Process QA for a single configuration."""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
                processed.add(key)
        print(f"Resuming: {len(processed)} entries already processed")
    
    with open(qg_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            if lang and data.get('lang_tgt', '') != lang:
                continue
            
            key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
            if key in processed:
                continue
            
            sentence = data.get(sentence_type, None)
            questions = data.get("questions", None)

            if sentence and questions:
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
                
                print(f"> {generated_answers[:80]}...")

                data['answers'] = generated_answers
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--qg_input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--sentence_type", type=str, default="src")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--pipeline", type=str, default="vanilla")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results Qwen3B baseline")

    if args.run_all:
        qg_file = os.path.join(results_dir, "QG", "biomqm", f"{args.pipeline}_qwen-3b.jsonl")
        
        if not os.path.exists(qg_file):
            print(f"QG file not found: {qg_file}")
            return
        
        for lang in LANGUAGES:
            print(f"\n{'='*60}")
            print(f"Processing SOURCE for {lang}")
            print(f"{'='*60}")
            
            output_file = os.path.join(results_dir, "QA", "biomqm", "source", f"{lang}-{args.pipeline}.jsonl")
            process_qa(tokenizer, model, device, qg_file, output_file, "src", lang)
            
            print(f"\n{'='*60}")
            print(f"Processing BT for {lang}")
            print(f"{'='*60}")
            
            output_file = os.path.join(results_dir, "QA", "biomqm", "bt", f"{lang}-{args.pipeline}.jsonl")
            process_qa(tokenizer, model, device, qg_file, output_file, "bt_tgt", lang)
        
        print(f"\n{'='*60}")
        print("BIOMQM QA completed!")
        print(f"{'='*60}")
    else:
        if not args.qg_input_path:
            args.qg_input_path = os.path.join(results_dir, "QG", "biomqm", f"{args.pipeline}_qwen-3b.jsonl")
        
        if not args.output_path:
            args.output_path = os.path.join(results_dir, "QA", "biomqm", args.sentence_type, f"{args.lang or 'all'}-{args.pipeline}.jsonl")
        
        process_qa(tokenizer, model, device, args.qg_input_path, args.output_path, args.sentence_type, args.lang)


if __name__ == "__main__":
    main()

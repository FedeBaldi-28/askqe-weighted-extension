import torch
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

vanilla = """Task: You will be given an English sentence. Your goal is to generate a list of questions that can be answered by the sentence. Output only the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: It's possible that 10 people in an 8 by 10 room could generate enough CO2 in 10 hours to get to that level.
Questions: ["What is possible?", "How many people could be in the room?", "What size is the room?", "What could the people generate?", "How long would it take?", "What level could be reached?"]

Sentence: Diabetes mellitus (784, 10.9%), chronic lung disease (656, 9.2%), and cardiovascular disease (647, 9.0%) were the most frequently reported conditions among all cases.
Questions: ["What were the most frequently reported conditions among all cases?", "Which conditions were reported with a frequency of 10.9%, 9.2%, and 9.0%, respectively?", "What percentage of cases reported diabetes mellitus?", "What percentage of cases reported chronic lung disease?", "What percentage of cases reported cardiovascular disease?"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: """


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
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str, default="vanilla")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_file = os.path.join(project_root, "biomqm", "dev_with_backtranslation.jsonl")
    
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = os.path.join(project_root, "results Qwen3B baseline", "QG", "biomqm")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.prompt}_qwen-3b.jsonl")

    # Track processed for resume
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
                processed.add(key)
        print(f"Resuming: {len(processed)} entries already processed")

    total_lines = 5216  # Total rows in dev_with_backtranslation.jsonl
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_path, 'a', encoding='utf-8') as f_out:
        for idx, line in enumerate(f_in, 1):
            print(f"Processing src {idx}/{total_lines}")
            data = json.loads(line)
            
            key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
            if key in processed:
                continue
            
            sentence = data.get('src', None)
            
            if sentence:
                prompt = vanilla.replace("{{sentence}}", sentence)

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
                generated_questions = tokenizer.decode(response, skip_special_tokens=True)
                
                if generated_questions:
                    generated_questions = generated_questions.strip('"\'')

                print(f"> {generated_questions[:80]}...")

                data['questions'] = generated_questions
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\nQG completed. Output: {output_path}")


if __name__ == "__main__":
    main()

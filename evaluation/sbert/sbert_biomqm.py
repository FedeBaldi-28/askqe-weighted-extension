import json
import nltk
import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

nltk.download("punkt")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


languages = ["de", "es", "fr", "ru", "zh-CN"]
SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results Qwen3B baseline")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def get_max_severity(errors_tgt):
    if not errors_tgt:
        return "Neutral"
    severity_order = {"Critical": 4, "Major": 3, "Minor": 2, "Neutral": 1}
    max_severity = "Neutral"
    max_order = 0
    for error in errors_tgt:
        sev = error.get("severity", "Neutral")
        if severity_order.get(sev, 0) > max_order:
            max_order = severity_order[sev]
            max_severity = sev
    return max_severity


for language in languages:
    print(f"\n{'='*60}")
    print(f"Processing: {language}")
    print(f"{'='*60}")
    
    predicted_file = os.path.join(results_dir, "QA", "biomqm", "bt", f"{language}-vanilla.jsonl")
    reference_file = os.path.join(results_dir, "QA", "biomqm", "source", f"{language}-vanilla.jsonl")
    
    # Load reference data
    ref_data_dict = {}
    try:
        with open(reference_file, "r", encoding="utf-8") as ref_file:
            for line in ref_file:
                data = json.loads(line)
                key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
                ref_data_dict[key] = data
    except FileNotFoundError as e:
        print(f"Reference file not found: {e}")
        continue
    
    results_by_severity = {sev: [] for sev in SEVERITIES}
    
    try:
        with open(predicted_file, "r", encoding="utf-8") as pred_file:
            for pred_line in pred_file:
                try:
                    pred_data = json.loads(pred_line)
                    key = f"{pred_data.get('src', '')}_{pred_data.get('lang_tgt', '')}"
                    ref_data = ref_data_dict.get(key, {})
                    
                    predicted_answers = pred_data.get("answers", [])
                    reference_answers = ref_data.get("answers", [])
                    severity = get_max_severity(pred_data.get("errors_tgt", []))

                    if isinstance(predicted_answers, str):
                        try:
                            predicted_answers = json.loads(predicted_answers)
                        except json.JSONDecodeError:
                            continue

                    if isinstance(reference_answers, str):
                        try:
                            reference_answers = json.loads(reference_answers)
                        except json.JSONDecodeError:
                            continue

                    if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                        continue
                    if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                        continue
                    
                    for pred, ref in zip(predicted_answers, reference_answers):
                        if not isinstance(pred, str) or not isinstance(ref, str):
                            continue
                        if pred.strip() == "" or ref.strip() == "":
                            continue

                        encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
                        encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

                        with torch.no_grad():
                            pred_output = model(**encoded_pred)
                            ref_output = model(**encoded_ref)

                        pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                        pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                        ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                        ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                        cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                        results_by_severity[severity].append(cos_sim)

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        continue

    print(f"\nResults by Severity:")
    print(f"{'Severity':<10} {'Count':>6} {'Avg CosSim':>12}")
    print("-" * 30)
    for sev in SEVERITIES:
        if results_by_severity[sev]:
            avg_sim = sum(results_by_severity[sev]) / len(results_by_severity[sev])
            print(f"{sev:<10} {len(results_by_severity[sev]):>6} {avg_sim:>12.3f}")

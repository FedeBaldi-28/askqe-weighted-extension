import json
import nltk
import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os

# Funzione per il Mean Pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

nltk.download("punkt", quiet=True)

# Configurazione Argomenti
parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True, help="Path al file CSV di output")
args = parser.parse_args()

# === CONFIGURAZIONE COMPLETA ===
languages = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["atomic", "semantic", "vanilla"]

perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                 "intensifier", "expansion_impact", "omission", "alteration"]

# Percorsi
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results Qwen3B baseline")

# === SETUP GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Caricamento Modello e Tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model.to(device) # Sposta il modello su GPU
model.eval()     # Imposta in modalità valutazione

# === ESECUZIONE ===
with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["language", "perturbation", "pipeline", "cosine_similarity", "num_comparison"])

    for language in languages:
        for pipeline in pipelines:
            for perturbation in perturbations:
                print(f"Processing: {language} | {pipeline} | {perturbation}")

                # Lettura da QA_FIXED
                predicted_file = os.path.join(results_dir, "QA_FIXED", "bt", f"{language}", f"{pipeline}", f"{language}-{pipeline}-{perturbation}.jsonl")
                reference_file = os.path.join(results_dir, "QA_FIXED", "source", f"en-{pipeline}.jsonl")

                total_cosine_similarity = 0
                num_comparisons = 0

                try:
                    with open(predicted_file, "r", encoding="utf-8") as pred_f, open(reference_file, "r", encoding="utf-8") as ref_f:
                        for pred_line, ref_line in zip(pred_f, ref_f):
                            try:
                                pred_data = json.loads(pred_line)
                                ref_data = json.loads(ref_line)

                                # Estrazione e pulizia dati
                                predicted_answers = pred_data.get("answers", [])
                                reference_answers = ref_data.get("answers", [])

                                # Fix per stringhe residue
                                if isinstance(predicted_answers, str):
                                    try: predicted_answers = json.loads(predicted_answers)
                                    except: predicted_answers = []
                                if isinstance(reference_answers, str):
                                    try: reference_answers = json.loads(reference_answers)
                                    except: reference_answers = []

                                if not isinstance(predicted_answers, list): predicted_answers = []
                                if not isinstance(reference_answers, list): reference_answers = []

                                # === LOGICA PADDING / TRUNCATING ===
                                len_p = len(predicted_answers)
                                len_r = len(reference_answers)

                                # 1. Se Source vuoto, salta la riga
                                if len_r == 0:
                                    continue

                                # 2. Se BT ha meno risposte, padding con stringhe vuote
                                if len_p < len_r:
                                    predicted_answers.extend([""] * (len_r - len_p))
                                # 3. Se BT ha più risposte, truncating
                                elif len_p > len_r:
                                    predicted_answers = predicted_answers[:len_r]
                                # ===================================

                                for pred, ref in zip(predicted_answers, reference_answers):
                                    # Normalizzazione a stringa
                                    if not isinstance(pred, str): pred = str(pred) if pred is not None else ""
                                    if not isinstance(ref, str): ref = str(ref) if ref is not None else ""
                                    
                                    pred = pred.strip()
                                    ref = ref.strip()

                                    # Se source è vuoto, skip
                                    if ref == "":
                                        continue
                                    
                                    # Se predizione è vuota (Padding), similarità 0
                                    if pred == "":
                                        total_cosine_similarity += 0.0
                                        num_comparisons += 1
                                        continue

                                    # Tokenizzazione e Spostamento su GPU
                                    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt').to(device)
                                    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt').to(device)

                                    with torch.no_grad():
                                        pred_output = model(**encoded_pred)
                                        ref_output = model(**encoded_ref)

                                    # Pooling e Normalizzazione
                                    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                                    pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                                    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                                    ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                                    # Calcolo similarità
                                    cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                                    total_cosine_similarity += cos_sim
                                    num_comparisons += 1

                            except json.JSONDecodeError:
                                continue

                except FileNotFoundError:
                    # print(f"File not found: {predicted_file}")
                    continue

                # Scrittura CSV per la configurazione corrente
                if num_comparisons > 0:
                    avg_cosine_similarity = total_cosine_similarity / num_comparisons
                    # print(f"-> Avg Cosine: {avg_cosine_similarity:.4f}")
                    csv_writer.writerow([language, perturbation, pipeline, avg_cosine_similarity, num_comparisons])
                else:
                    # Se il file era vuoto o rotto, scriviamo 0 o N/A per tenere traccia? 
                    # Lo script originale non scriveva nulla, mantengo così.
                    pass

print("✅ Calcolo completato.")
"""
String Comparison Evaluation for BioMQM Dataset

Questo script calcola le metriche di similarità tra le risposte generate
dal SOURCE (inglese originale) e quelle generate dalla BACKTRANSLATION.
Le metriche sono: F1, Exact Match, chrF, BLEU.

Adattato da string_comparison.py per CONTRATICO al dataset BIOMQM.
"""

import json
import nltk
import os
from utils import compare_answers  # Importa le funzioni di confronto da utils.py

nltk.download("punkt")

# ========================================
# CONFIGURAZIONE BIOMQM
# ========================================

# Le 5 lingue presenti nel dataset BIOMQM
languages = ["de", "es", "fr", "ru", "zh-CN"]

# BIOMQM usa solo il pipeline "vanilla" (no atomic/semantic per ora)
pipelines = ["vanilla"]

# BIOMQM non ha perturbations come CONTRATICO, quindi lista vuota
# Nel nostro caso confrontiamo direttamente source vs bt
perturbations = [""]  # Placeholder per mantenere la struttura

# ========================================
# PERCORSI FILES
# ========================================

# Percorso relativo dalla posizione dello script
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
project_root = os.path.dirname(os.path.dirname(script_dir))

# Percorso per Colab/Kaggle - modifica se necessario
results_dir = "/content/askqe/results Qwen3B baseline"

# ========================================
# COSTANTI PER SEVERITY
# ========================================

# Le 4 severity possibili nel dataset BIOMQM
SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def get_max_severity(errors_tgt):
    """
    Estrae la severity massima dalla lista di errori.
    
    Nel dataset BIOMQM, ogni riga può avere più errori.
    Questa funzione trova l'errore con severity più alta.
    
    Ordine di priorità: Critical > Major > Minor > Neutral
    
    Args:
        errors_tgt: lista di errori con campo 'severity'
        
    Returns:
        str: la severity massima trovata (default: "Neutral")
    """
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


# ========================================
# LOOP PRINCIPALE
# ========================================

for language in languages:
    for pipeline in pipelines:
        # DIFFERENZA DA CONTRATICO: 
        # BIOMQM non ha perturbations, quindi usiamo direttamente i file QA
        
        # File con le risposte dalla BACKTRANSLATION (da valutare)
        predicted_file = os.path.join(results_dir, "QA", "biomqm", "bt", f"{language}-{pipeline}.jsonl")
        
        # File con le risposte dal SOURCE (reference/gold standard)
        reference_file = os.path.join(results_dir, "QA", "biomqm", "source", f"{language}-{pipeline}.jsonl")

        results_list = []
        
        # Skip se manca il file
        if not os.path.exists(predicted_file) or not os.path.exists(reference_file):
            print(f"File mancante per {language}-{pipeline}")
            continue

        # ========================================
        # CARICAMENTO REFERENCE IN DIZIONARIO
        # ========================================
        # DIFFERENZA DA CONTRATICO:
        # BIOMQM può avere ordini diversi tra source e bt,
        # quindi usiamo un dizionario per il matching
        
        ref_data_dict = {}
        try:
            with open(reference_file, "r", encoding="utf-8") as ref_file:
                for line in ref_file:
                    data = json.loads(line)
                    # Chiave unica: combinazione di src + lingua
                    key = f"{data.get('src', '')}_{data.get('lang_tgt', '')}"
                    ref_data_dict[key] = data
        except FileNotFoundError as e:
            print(f"Reference file not found: {e}")
            continue
        
        # ========================================
        # ELABORAZIONE FILE PREDICTED (BT)
        # ========================================
        
        try:
            with open(predicted_file, "r", encoding="utf-8") as pred_file:
                for pred_line in pred_file:
                    try:
                        pred_data = json.loads(pred_line)
                        
                        # Trova la reference corrispondente usando la chiave
                        key = f"{pred_data.get('src', '')}_{pred_data.get('lang_tgt', '')}"
                        ref_data = ref_data_dict.get(key, {})
                        
                        # Estrai le risposte
                        predicted_answers = pred_data.get("answers", [])
                        reference_answers = ref_data.get("answers", [])
                        
                        # DIFFERENZA DA CONTRATICO:
                        # Estraiamo la severity dagli errori per poter
                        # calcolare metriche separate per ogni severity level
                        severity = get_max_severity(pred_data.get("errors_tgt", []))

                        # === SAFETY CHECKS ===
                        # Le risposte potrebbero essere stringhe JSON invece di liste
                        if isinstance(predicted_answers, str):
                            try: predicted_answers = json.loads(predicted_answers)
                            except: predicted_answers = []
                        if isinstance(reference_answers, str):
                            try: reference_answers = json.loads(reference_answers)
                            except: reference_answers = []

                        if not isinstance(predicted_answers, list): predicted_answers = []
                        if not isinstance(reference_answers, list): reference_answers = []

                        # === LOGICA PADDING & TRUNCATING ===
                        # Gestisce i casi in cui il numero di risposte è diverso
                        len_p = len(predicted_answers)
                        len_r = len(reference_answers)

                        # 1. Se SOURCE non ha risposte, nulla da valutare
                        if len_r == 0:
                            continue

                        # 2. Se BT ha meno risposte, aggiungiamo padding (stringhe vuote)
                        #    Queste prenderanno score 0 nel confronto
                        if len_p < len_r:
                            predicted_answers.extend([""] * (len_r - len_p))
                        
                        # 3. Se BT ha più risposte, tagliamo l'eccesso
                        elif len_p > len_r:
                            predicted_answers = predicted_answers[:len_r]
                        
                        # Ora len_p == len_r, possiamo confrontare

                        # ========================================
                        # CALCOLO METRICHE PER OGNI COPPIA RISPOSTA
                        # ========================================
                        
                        row_scores = []
                        num_mismatches = 0  # AGGIUNTO: conta quante risposte sono diverse
                        
                        for pred, ref in zip(predicted_answers, reference_answers):
                            # Converti non-stringhe in stringhe
                            if not isinstance(pred, str):
                                pred = str(pred) if pred is not None else ""
                            if not isinstance(ref, str):
                                ref = str(ref) if ref is not None else ""
                            
                            # IMPORTANTE: Saltiamo SOLO se la REFERENCE è vuota
                            # Se 'pred' è vuoto (padding), deve passare e prendere 0
                            if not ref.strip():
                                continue
                            
                            # Calcola le 4 metriche usando utils.compare_answers
                            f1, EM, chrf, bleu = compare_answers(pred, ref)
                            row_scores.append({
                                "f1": f1,
                                "em": EM,
                                "chrf": chrf,
                                "bleu": bleu
                            })
                            
                            # Se Exact Match è 0, conta come mismatch
                            if EM == 0:
                                num_mismatches += 1

                        # ========================================
                        # SALVATAGGIO RISULTATO PER RIGA
                        # ========================================
                        
                        # DIFFERENZA DA CONTRATICO:
                        # Aggiungiamo severity e num_mismatches per decision rule
                        row_data = {
                            "src": pred_data.get("src", ""),
                            "lang_tgt": pred_data.get("lang_tgt", ""),
                            "severity": severity,  # AGGIUNTO
                            "num_mismatches": num_mismatches,  # AGGIUNTO
                            "scores": row_scores
                        }
                        results_list.append(row_data)

                    except json.JSONDecodeError as e:
                        print(f"Skipping corrupted line: {e}")
                        continue

        except FileNotFoundError as e:
            print(f"File not found during open: {e}")

        # ========================================
        # OUTPUT: SALVATAGGIO RISULTATI
        # ========================================
        
        # Percorso output specifico per BIOMQM
        jsonl_output_file = os.path.join(
            results_dir, 
            "evaluation", 
            "string-comparison", 
            "biomqm",  # Sottocartella specifica per BIOMQM
            f"{language}-{pipeline}.jsonl"
        )
        
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)
        with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
            for row in results_list:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        # ========================================
        # STAMPA SUMMARY PER SEVERITY
        # ========================================
        
        print(f"\n{'='*50}")
        print(f"Language: {language} | Pipeline: {pipeline}")
        print(f"{'='*50}")
        
        for sev in SEVERITIES:
            sev_rows = [r for r in results_list if r["severity"] == sev]
            if sev_rows:
                # Calcola medie per questa severity
                total_f1 = 0
                total_em = 0
                total_bleu = 0
                count = 0
                
                for r in sev_rows:
                    if r["scores"]:
                        total_f1 += sum(s["f1"] for s in r["scores"]) / len(r["scores"])
                        total_em += sum(s["em"] for s in r["scores"]) / len(r["scores"])
                        total_bleu += sum(s["bleu"] for s in r["scores"]) / len(r["scores"])
                        count += 1
                
                if count > 0:
                    avg_f1 = total_f1 / count
                    avg_em = total_em / count
                    avg_bleu = total_bleu / count
                    print(f"  {sev:<10}: {len(sev_rows):>4} rows | F1: {avg_f1:.3f} | EM: {avg_em:.3f} | BLEU: {avg_bleu:.3f}")
        
        print(f"\nSaved: {jsonl_output_file}")

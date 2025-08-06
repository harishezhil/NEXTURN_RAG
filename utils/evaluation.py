"""
WHY evaluation.py?
----------

Evaluation utilities for comparing LLM-generated answers to ground truth — 
to know if the AI is just confidently wrong.

Includes:
---------
- ROUGE-1 and ROUGE-L: Because word overlap still tells us something.
- Cosine Similarity: Checks if two answers "feel" the same in embedding space.
- Token-level F1: Measures how much your prediction and truth actually agree, word for word.
- Accuracy: If token-F1 ≥ 0.6, we pretend it's correct. Can be adjusted.

Functions:
----------
- tokenize(text): Turns a sentence into a set of lowercase word tokens. Punctuation? Gone.
- compute_token_f1(ref, pred): Computes precision, recall, and F1 score between token sets.
- evaluate_predictions(ground_truth, predictions):
    Takes two dicts (question → answer), compares each pair, and returns a metrics summary.
"""


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import numpy as np

import re

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def tokenize(text):
    # Simple word tokenizer
    return set(re.findall(r'\w+', text.lower()))

def compute_token_f1(reference, prediction):
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    common = ref_tokens.intersection(pred_tokens)
    if not common:
        return 0.0, 0.0, 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

def evaluate_predictions(ground_truth: dict, predictions: dict):
    rouge1_list, rougel_list, cosine_list, token_f1_list = [], [], [], []
    correct_count = 0

    for question in ground_truth:
        ref = ground_truth[question].strip()
        pred = predictions.get(question, "").strip()

        # ROUGE
        scores = scorer.score(ref, pred)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rougel_list.append(scores["rougeL"].fmeasure)

        # Cosine similarity
        emb_ref = model.encode([ref])[0]
        emb_pred = model.encode([pred])[0]
        cosine_sim = cosine_similarity([emb_ref], [emb_pred])[0][0]
        cosine_list.append(cosine_sim)

        # Token F1
        _, _, token_f1 = compute_token_f1(ref, pred)
        token_f1_list.append(token_f1)

        # Accuracy logic: consider correct if token_f1 ≥ 0.6 (you can adjust threshold)
        if token_f1 >= 0.6:
            correct_count += 1
            

    total = len(ground_truth)
    accuracy = correct_count / total if total > 0 else 0

    return {
        "ROUGE-1": round(np.mean(rouge1_list), 4),
        "ROUGE-L": round(np.mean(rougel_list), 4),
        "Cosine Similarity": round(np.mean(cosine_list), 4),
        "F1_Score": round(np.mean(token_f1_list), 4),
        "Accuracy": round(accuracy, 4)
    }

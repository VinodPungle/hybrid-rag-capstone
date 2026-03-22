"""
LLMOps Evaluation Metrics (Step 8)

Provides BLEU, ROUGE, and faithfulness scoring to measure
RAG answer quality. These metrics are standard in NLP/LLMOps:

  - BLEU (1-4): Measures n-gram overlap between generated and reference answers.
    Higher = more similar. BLEU-4 uses 4-gram overlap and is the strictest.

  - ROUGE (1, 2, L): Recall-oriented metrics measuring overlap.
    ROUGE-1 = unigram, ROUGE-2 = bigram, ROUGE-L = longest common subsequence.

  - Faithfulness: Estimates whether the generated answer is grounded in the
    source context (i.e., not hallucinated). Returns a score from 0 to 1.

Which metrics are enabled is controlled by config.yaml → llmops.evaluation.metrics.
"""

import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# [Step 1] Import config loader to check which metrics are enabled
from config.settings import get as cfg

# [Step 2] Import structured logger
from utils.logger import get_logger

logger = get_logger(__name__)


def bleu_score(reference: str, candidate: str) -> dict:
    """
    Compute BLEU score between a reference and candidate answer.

    BLEU measures how many n-grams in the candidate appear in the reference.
    Uses smoothing to handle cases where higher-order n-grams have zero matches.

    Args:
        reference: The expected/gold answer
        candidate: The generated answer from the RAG pipeline

    Returns:
        Dict with bleu_1 (unigram), bleu_2, bleu_3, bleu_4 (4-gram) scores.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Return zeros if either text is empty
    if not ref_tokens or not cand_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Smoothing prevents zero scores when some n-gram counts are zero
    smooth = SmoothingFunction().method1

    scores = {
        "bleu_1": sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu_2": sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "bleu_3": sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        "bleu_4": sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

    return scores


def rouge_scores(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    ROUGE is recall-oriented: measures how much of the reference
    content is captured by the candidate.

    Args:
        reference: The expected/gold answer
        candidate: The generated answer from the RAG pipeline

    Returns:
        Dict with rouge1, rouge2, rougeL, each containing
        precision, recall, and fmeasure (F1 score).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = scorer.score(reference, candidate)

    return {
        key: {
            "precision": round(val.precision, 4),
            "recall": round(val.recall, 4),
            "fmeasure": round(val.fmeasure, 4),
        }
        for key, val in results.items()
    }


def faithfulness_score(answer: str, context: str) -> dict:
    """
    Estimate faithfulness by checking how many claims in the answer
    are grounded in the source context (i.e., not hallucinated).

    Approach: Split the answer into sentences, then for each sentence,
    check if at least 50% of its meaningful words (4+ chars) appear
    in the source context. A sentence is "grounded" if it passes this check.

    Args:
        answer: The generated answer from the RAG pipeline
        context: The source context that was used for generation

    Returns:
        Dict with:
          - score: 0.0 to 1.0 (fraction of grounded sentences)
          - grounded: number of sentences grounded in context
          - total: total number of sentences evaluated
    """
    # Split answer into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return {"score": 1.0, "grounded": 0, "total": 0}

    context_lower = context.lower()
    grounded = 0

    for sentence in sentences:
        # Extract meaningful words (4+ chars) — filters out short common words
        words = [w for w in re.findall(r'\b\w+\b', sentence.lower()) if len(w) >= 4]
        if not words:
            grounded += 1
            continue

        # Check what fraction of meaningful words appear in context
        matches = sum(1 for w in words if w in context_lower)
        overlap = matches / len(words)

        # Consider the sentence "grounded" if >= 50% of its words are in context
        if overlap >= 0.5:
            grounded += 1

    score = grounded / len(sentences) if sentences else 1.0

    return {
        "score": round(score, 4),
        "grounded": grounded,
        "total": len(sentences),
    }


def evaluate(reference: str, candidate: str, context: str = "") -> dict:
    """
    Run all enabled evaluation metrics on a candidate answer.

    Which metrics run is controlled by config.yaml → llmops.evaluation.metrics.
    Possible values: "bleu", "rouge", "faithfulness".

    Args:
        reference: The expected/gold answer
        candidate: The generated answer from the RAG pipeline
        context: The source context used for generation (needed for faithfulness)

    Returns:
        Dict with scores for each enabled metric.
    """
    # Read which metrics are enabled from config.yaml
    eval_cfg = cfg("llmops", "evaluation") or {}
    enabled_metrics = eval_cfg.get("metrics", [])
    results = {}

    if "bleu" in enabled_metrics:
        results["bleu"] = bleu_score(reference, candidate)

    if "rouge" in enabled_metrics:
        results["rouge"] = rouge_scores(reference, candidate)

    if "faithfulness" in enabled_metrics and context:
        results["faithfulness"] = faithfulness_score(candidate, context)

    logger.info("Evaluation complete: %s", list(results.keys()))
    return results

from evaluation.metrics import bleu_score, rouge_scores, faithfulness_score, evaluate


def test_bleu_identical():
    """Identical texts should have high BLEU scores."""
    text = "The audit committee oversees financial reporting."
    scores = bleu_score(text, text)
    assert scores["bleu_1"] > 0.9
    assert scores["bleu_4"] > 0.9


def test_bleu_different():
    """Completely different texts should have low BLEU scores."""
    scores = bleu_score("The cat sat on the mat", "Dogs run in the park quickly")
    assert scores["bleu_4"] < 0.2


def test_bleu_empty():
    """Empty inputs should return zero scores."""
    scores = bleu_score("", "some text")
    assert scores["bleu_1"] == 0.0


def test_rouge_identical():
    """Identical texts should have perfect ROUGE scores."""
    text = "The audit committee oversees financial reporting."
    scores = rouge_scores(text, text)
    assert scores["rouge1"]["fmeasure"] == 1.0
    assert scores["rougeL"]["fmeasure"] == 1.0


def test_rouge_partial():
    """Partially overlapping texts should have mid-range scores."""
    ref = "The audit committee reviews internal controls and compliance."
    cand = "The audit committee is responsible for oversight of internal controls."
    scores = rouge_scores(ref, cand)
    assert 0.2 < scores["rouge1"]["fmeasure"] < 0.9


def test_faithfulness_grounded():
    """Answer grounded in context should have high faithfulness."""
    context = "The audit committee reviews financial statements and internal controls."
    answer = "The audit committee reviews financial statements. It also reviews internal controls."
    result = faithfulness_score(answer, context)
    assert result["score"] >= 0.8


def test_faithfulness_ungrounded():
    """Answer with claims not in context should have low faithfulness."""
    context = "The company was founded in 1990."
    answer = "The audit committee reviews financial statements. It oversees compliance and risk management processes."
    result = faithfulness_score(answer, context)
    assert result["score"] < 0.5


def test_evaluate_all_metrics():
    """evaluate() should return results for all enabled metrics."""
    ref = "The committee oversees financial reporting."
    cand = "The committee is responsible for overseeing financial reporting processes."
    ctx = "The committee oversees financial reporting and internal controls."

    results = evaluate(ref, cand, context=ctx)
    assert "bleu" in results
    assert "rouge" in results
    assert "faithfulness" in results
    assert results["bleu"]["bleu_1"] > 0
    assert results["rouge"]["rougeL"]["fmeasure"] > 0
    assert results["faithfulness"]["score"] > 0

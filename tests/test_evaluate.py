from src.evaluate import Query, dcg, evaluate, ndcg_at_k


class FakeHit:
    def __init__(self, url):
        self.document = type("D", (), {"url": url})()


class FakeService:
    def __init__(self, order):
        self.order = order

    def search(self, query, *, limit=10, now=None):
        return [FakeHit(url) for url in self.order[:limit]]


QUERY = Query(text="q", relevance={"a": 2, "b": 2, "c": 1})


def test_perfect_ranking_scores_one():
    assert ndcg_at_k(["a", "b", "c"], QUERY, 10) == 1.0


def test_reversed_ranking_scores_below_perfect():
    assert ndcg_at_k(["c", "b", "a"], QUERY, 10) < 1.0


def test_irrelevant_results_score_zero():
    assert ndcg_at_k(["x", "y", "z"], QUERY, 10) == 0.0


def test_position_matters():
    """The property precision@k does not have, and the reason nDCG is primary."""
    early = ndcg_at_k(["a", "x", "x", "x", "x"], QUERY, 10)
    late = ndcg_at_k(["x", "x", "x", "x", "a"], QUERY, 10)
    assert early > late


def test_grade_two_outweighs_grade_one():
    assert ndcg_at_k(["a"], QUERY, 1) > ndcg_at_k(["c"], QUERY, 1)


def test_query_with_no_relevant_documents_scores_zero_not_nan():
    """A gap in the labels must not produce NaN and poison the mean."""
    empty = Query(text="q", relevance={})
    assert ndcg_at_k(["a"], empty, 10) == 0.0


def test_dcg_discounts_later_positions():
    assert dcg([2, 0]) > dcg([0, 2])


def test_k_truncates_the_result_list():
    assert ndcg_at_k(["x", "x", "a"], QUERY, 2) == 0.0


def test_report_aggregates_across_queries():
    report = evaluate(FakeService(["a", "b", "c"]), [QUERY, QUERY], k=10)
    assert report.n_queries == 2 and report.ndcg == 1.0


def test_mrr_uses_the_first_relevant_position():
    report = evaluate(FakeService(["x", "x", "a"]), [QUERY], k=10)
    assert abs(report.mrr - 1 / 3) < 1e-9


def test_recall_is_relative_to_the_labelled_set():
    report = evaluate(FakeService(["a", "b", "c"]), [QUERY], k=10)
    assert report.recall == 1.0

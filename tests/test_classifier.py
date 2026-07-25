"""Unit tests for the classical ML stack."""

from __future__ import annotations

import math

import pytest

from ticketiq.ml.classifier import TicketClassifier, train_and_evaluate
from ticketiq.ml.dataset import LabelledTicket, load_dataset, stratified_split
from ticketiq.ml.logistic_regression import SoftmaxRegression
from ticketiq.ml.metrics import evaluate
from ticketiq.ml.text import featurize, tokenize
from ticketiq.ml.vectorizer import TfidfVectorizer


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
def test_tokenize_lowercases_and_drops_stopwords() -> None:
    tokens = tokenize("The Billing Invoice was WRONG")
    assert "the" not in tokens
    assert "billing" in tokens and "invoice" in tokens and "wrong" in tokens


def test_featurize_emits_bigrams() -> None:
    features = featurize("double charged again", ngram_range=(1, 2))
    assert "double_charged" in features
    assert "double" in features


# --------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------
def test_idf_is_smoothed_and_monotonic() -> None:
    docs = ["refund refund", "refund login", "login error", "login error crash"]
    vec = TfidfVectorizer(ngram_range=(1, 1)).fit(docs)
    idf_refund = vec.idf_[vec.vocabulary_["refund"]]
    idf_login = vec.idf_[vec.vocabulary_["login"]]
    idf_crash = vec.idf_[vec.vocabulary_["crash"]]
    # 'login' appears in 3 docs, 'refund' in 2, 'crash' in 1 -> rarer means higher idf.
    assert idf_crash > idf_refund > idf_login
    assert idf_login >= 1.0  # smoothing keeps idf strictly positive


def test_idf_matches_closed_form() -> None:
    docs = ["alpha beta", "alpha gamma", "delta"]
    vec = TfidfVectorizer(ngram_range=(1, 1)).fit(docs)
    expected = math.log((1 + 3) / (1 + 2)) + 1.0  # 'alpha' has df=2, n=3
    assert vec.idf_[vec.vocabulary_["alpha"]] == pytest.approx(expected)


def test_transform_is_l2_normalised() -> None:
    docs = ["refund invoice billing", "login sso password"]
    vec = TfidfVectorizer(ngram_range=(1, 1)).fit(docs)
    out = vec.transform_one(docs[0])
    assert math.sqrt(sum(v * v for v in out.values())) == pytest.approx(1.0)


def test_unknown_terms_are_ignored_not_crashing() -> None:
    vec = TfidfVectorizer().fit(["refund invoice"])
    assert vec.transform_one("completely unrelated vocabulary") == {}


def test_min_df_filters_rare_terms() -> None:
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2).fit(
        ["refund refund", "refund login", "login"]
    )
    assert "refund" in vec.vocabulary_
    assert vec.n_features == 2


def test_vectorizer_roundtrips_through_dict() -> None:
    vec = TfidfVectorizer().fit(["refund invoice billing", "login sso"])
    restored = TfidfVectorizer.from_dict(vec.to_dict())
    assert restored.vocabulary_ == vec.vocabulary_
    assert restored.transform_one("refund invoice") == vec.transform_one("refund invoice")


# --------------------------------------------------------------------------
# Logistic regression (softmax)
# --------------------------------------------------------------------------
def test_logreg_separates_two_obvious_classes() -> None:
    docs = [
        "refund invoice billing charge",
        "invoice billing payment refund",
        "login password sso locked",
        "password reset login access",
    ]
    labels = ["billing", "billing", "account", "account"]
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(docs)
    model = SoftmaxRegression().fit(X, labels, vec.n_features)

    assert model.predict_one(vec.transform_one("refund my invoice"))[0] == "billing"
    assert model.predict_one(vec.transform_one("password reset"))[0] == "account"


def test_logreg_predict_proba_is_a_distribution() -> None:
    docs = ["refund invoice", "login password", "crash error"]
    labels = ["billing", "account", "technical"]
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(docs)
    model = SoftmaxRegression().fit(X, labels, vec.n_features)
    proba = model.predict_proba_one(vec.transform_one("refund"))
    assert sum(proba.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in proba.values())


def test_logreg_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        SoftmaxRegression().fit([], [], 0)
    with pytest.raises(ValueError):
        SoftmaxRegression().fit([{0: 1.0}], ["a", "b"], 1)


def test_logreg_ignores_unseen_features() -> None:
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(["refund invoice", "login password"])
    model = SoftmaxRegression().fit(X, ["billing", "account"], vec.n_features)
    # A feature index past the training vocabulary must not affect inference.
    label, confidence, _ = model.predict_one({9999: 1.0})
    assert label in ("billing", "account")
    assert 0.0 <= confidence <= 1.0


def test_logreg_roundtrips_through_dict() -> None:
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(["refund invoice", "login password"])
    model = SoftmaxRegression().fit(X, ["billing", "account"], vec.n_features)
    restored = SoftmaxRegression.from_dict(model.to_dict())
    query = vec.transform_one("refund")
    assert restored.predict_proba_one(query) == model.predict_proba_one(query)


def test_logreg_gradient_descent_reduces_the_loss() -> None:
    # The optimiser must actually learn: more epochs should not increase the
    # negative log-likelihood on the training data (convexity + gradient descent).
    docs = ["refund invoice billing", "login password sso", "crash error timeout"]
    labels = ["billing", "account", "technical"]
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(docs)

    def train_nll(epochs: int) -> float:
        model = SoftmaxRegression(epochs=epochs).fit(X, labels, vec.n_features)
        return -sum(math.log(model.predict_proba_one(x)[y]) for x, y in zip(X, labels))

    assert train_nll(200) < train_nll(2)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_metrics_on_a_hand_computed_example() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    report = evaluate(y_true, y_pred)
    assert report.accuracy == pytest.approx(0.75)
    # class a: tp=1 fp=0 fn=1 -> precision 1.0, recall 0.5, f1 2/3
    assert report.per_class["a"].precision == pytest.approx(1.0)
    assert report.per_class["a"].recall == pytest.approx(0.5)
    assert report.per_class["a"].f1 == pytest.approx(2 / 3)
    # class b: tp=2 fp=1 fn=0 -> precision 2/3, recall 1.0
    assert report.per_class["b"].precision == pytest.approx(2 / 3)
    assert report.per_class["b"].recall == pytest.approx(1.0)


def test_metrics_handle_a_never_predicted_class() -> None:
    report = evaluate(["a", "b"], ["a", "a"], labels=["a", "b", "c"])
    assert report.per_class["c"].precision == 0.0
    assert report.per_class["c"].f1 == 0.0
    assert report.per_class["b"].recall == 0.0


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate(["a"], ["a", "b"])


# --------------------------------------------------------------------------
# Dataset and end-to-end training
# --------------------------------------------------------------------------
def test_stratified_split_preserves_class_balance() -> None:
    tickets = [
        LabelledTicket(subject=f"s{i}", body="b", category="billing" if i % 2 else "technical")
        for i in range(40)
    ]
    train, test = stratified_split(tickets, test_size=0.25, seed=3)
    assert len(test) == 10
    assert len({t.category for t in test}) == 2
    assert not set(id(t) for t in train) & set(id(t) for t in test)


def test_split_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError):
        stratified_split([LabelledTicket("s", "b", "billing")], test_size=1.5)


def test_trained_classifier_beats_a_useful_threshold(isolated_env) -> None:
    _, report = train_and_evaluate(test_size=0.25, seed=1)
    # Chance is 0.25 on four balanced classes; anything below 0.85 means the
    # feature pipeline has regressed.
    assert report.accuracy > 0.85
    assert report.macro_f1 > 0.85


def test_classifier_predicts_sensible_categories(isolated_env) -> None:
    from ticketiq.config import get_settings

    tickets = load_dataset(get_settings().dataset_path)
    clf = TicketClassifier().fit(tickets)
    assert clf.predict("Double charged", "Refund the duplicate invoice").category == "billing"
    assert clf.predict("Dashboard crashes", "The app crashes and times out").category == "technical"
    assert clf.predict("Add dark mode", "Would be great to have a dark theme").category == (
        "feature_request"
    )


def test_classifier_persistence_roundtrip(isolated_env, tmp_path) -> None:
    from ticketiq.config import get_settings

    tickets = load_dataset(get_settings().dataset_path)
    clf = TicketClassifier().fit(tickets)
    path = tmp_path / "model.json"
    clf.save(path)
    restored = TicketClassifier.load(path)
    original = clf.predict("Double charged", "Refund please")
    reloaded = restored.predict("Double charged", "Refund please")
    assert original.category == reloaded.category
    assert original.confidence == pytest.approx(reloaded.confidence)


def test_predicting_before_training_raises() -> None:
    with pytest.raises(RuntimeError):
        TicketClassifier().predict("subject", "body")

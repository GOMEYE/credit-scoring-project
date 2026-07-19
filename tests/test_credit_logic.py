import sys
import os
import pandas as pd
import joblib

# Allow importing from the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def credit_score_from_probability(prob_good):
    """Mirrors the credit score calculation used in app.py"""
    return int(300 + (prob_good * 550))


def decision_from_score(score):
    """Mirrors the approval decision logic used in app.py"""
    if score >= 700:
        return "Approved"
    elif score >= 550:
        return "Conditional"
    else:
        return "Denied"


def confidence_from_probability(prob_good):
    """Mirrors the confidence calculation used in app.py"""
    return abs(prob_good - 0.5) * 2 * 100


class TestCreditScoreCalculation:
    def test_minimum_probability_gives_minimum_score(self):
        assert credit_score_from_probability(0.0) == 300

    def test_maximum_probability_gives_maximum_score(self):
        assert credit_score_from_probability(1.0) == 850

    def test_midpoint_probability(self):
        assert credit_score_from_probability(0.5) == 575

    def test_score_is_always_in_valid_range(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            score = credit_score_from_probability(p)
            assert 300 <= score <= 850


class TestDecisionThresholds:
    def test_high_score_is_approved(self):
        assert decision_from_score(700) == "Approved"
        assert decision_from_score(850) == "Approved"

    def test_mid_score_is_conditional(self):
        assert decision_from_score(550) == "Conditional"
        assert decision_from_score(699) == "Conditional"

    def test_low_score_is_denied(self):
        assert decision_from_score(549) == "Denied"
        assert decision_from_score(300) == "Denied"

    def test_boundary_values(self):
        # Exact threshold boundaries should not be misclassified
        assert decision_from_score(699) != "Approved"
        assert decision_from_score(700) != "Conditional"
        assert decision_from_score(549) != "Conditional"
        assert decision_from_score(550) != "Denied"


class TestConfidenceCalculation:
    def test_fifty_fifty_is_zero_confidence(self):
        assert confidence_from_probability(0.5) == 0.0

    def test_certain_good_is_full_confidence(self):
        assert confidence_from_probability(1.0) == 100.0

    def test_certain_bad_is_full_confidence(self):
        assert confidence_from_probability(0.0) == 100.0

    def test_confidence_is_symmetric(self):
         # 0.3 and 0.7 are equally far from 0.5, should give equal confidence
         # (using approximate equality due to floating-point precision)
         diff = abs(confidence_from_probability(0.3) - confidence_from_probability(0.7))
         assert diff < 0.0001

    def test_confidence_always_non_negative(self):
         for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
               assert confidence_from_probability(p) >= 0


class TestModelArtifacts:
    """Sanity checks that required model files exist and load correctly."""

    def test_xgboost_model_file_exists(self):
        path = os.path.join(MODELS_DIR, "xgboost_balanced_credit_model.pkl")
        assert os.path.exists(path), f"Model file not found: {path}"

    def test_xgboost_model_loads(self):
        path = os.path.join(MODELS_DIR, "xgboost_balanced_credit_model.pkl")
        model = joblib.load(path)
        assert model is not None

    def test_encoders_exist(self):
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]:
            path = os.path.join(MODELS_DIR, f"{col}_encoder.pkl")
            assert os.path.exists(path), f"Encoder not found: {path}"

    def test_sex_encoder_has_expected_classes(self):
        path = os.path.join(MODELS_DIR, "Sex_encoder.pkl")
        encoder = joblib.load(path)
        assert set(encoder.classes_) == {"male", "female"}


class TestModelPrediction:
    """End-to-end sanity check: does the model actually produce valid output?"""

    def test_prediction_on_sample_input(self):
        model_path = os.path.join(MODELS_DIR, "xgboost_balanced_credit_model.pkl")
        model = joblib.load(model_path)

        sex_encoder = joblib.load(os.path.join(MODELS_DIR, "Sex_encoder.pkl"))
        housing_encoder = joblib.load(os.path.join(MODELS_DIR, "Housing_encoder.pkl"))
        saving_encoder = joblib.load(os.path.join(MODELS_DIR, "Saving accounts_encoder.pkl"))
        checking_encoder = joblib.load(os.path.join(MODELS_DIR, "Checking account_encoder.pkl"))

        sample_input = pd.DataFrame({
            "Age": [30],
            "Sex": [sex_encoder.transform(["male"])[0]],
            "Job": [1],
            "Housing": [housing_encoder.transform(["own"])[0]],
            "Saving accounts": [saving_encoder.transform(["little"])[0]],
            "Checking account": [checking_encoder.transform(["little"])[0]],
            "Credit amount": [1000],
            "Duration": [12]
        })

        prediction = model.predict(sample_input)[0]
        probabilities = model.predict_proba(sample_input)[0]

        assert prediction in [0, 1]
        assert len(probabilities) == 2
        assert 0.0 <= probabilities[0] <= 1.0
        assert 0.0 <= probabilities[1] <= 1.0
        assert abs(probabilities[0] + probabilities[1] - 1.0) < 0.0001

      # Run this: pytest tests/ -v
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.main import app
from src.config import FEATURES_PATH, MODEL_INFO_PATH, MODEL_PATH, ROOT, FEATURE_SET
from src.data import load_data
from src.schema import TARGET, get_feature_columns


@pytest.fixture
def client():
    return app.test_client()


def test_saved_schema_matches_pipeline():
    pipeline = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    assert "smote" not in pipeline.named_steps
    assert list(pipeline.feature_names_in_) == [item["csv_name"] for item in features]
    assert all(item["csv_name"] != "Target" for item in features)


def test_no_resampling():
    pipeline = joblib.load(MODEL_PATH)
    assert "smote" not in pipeline.named_steps
    assert "smotenc" not in pipeline.named_steps
    assert list(pipeline.named_steps.keys()) == ["prep", "rf"]


def test_validation_rejects_missing_and_extra_fields(client):
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    
    # Missing required field
    partial = {item["id"]: 0 for item in features[:10]}
    response = client.post("/predict", json={"features": partial})
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_failed"

    # Extra unexpected field
    full = {}
    for item in features:
        full[item["id"]] = item["options"][0]["value"] if "options" in item else item.get("median", item.get("observed_min", 0))
    full["extra_fake_column"] = 999
    response = client.post("/predict", json={"features": full})
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_failed"


def test_validation_rejects_out_of_bounds_and_unknown_options(client):
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    valid_sample = {}
    for item in features:
        valid_sample[item["id"]] = item["options"][0]["value"] if "options" in item else item.get("median", item.get("observed_min", 0))

    # Test out of bounds number (e.g. age = 150)
    sample_age_bad = dict(valid_sample)
    sample_age_bad["age_at_enrollment"] = 150
    resp = client.post("/predict", json={"features": sample_age_bad})
    assert resp.status_code == 400

    # Test unknown select option
    sample_opt_bad = dict(valid_sample)
    sample_opt_bad["course"] = 999999
    resp = client.post("/predict", json={"features": sample_opt_bad})
    assert resp.status_code == 400


def test_probabilities_follow_model_classes(client):
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    values = {}
    for item in features:
        values[item["id"]] = item["options"][0]["value"] if "options" in item else item.get("median", item.get("observed_min", 0))
    response = client.post("/predict", json={"features": values})
    assert response.status_code == 200
    body = response.get_json()
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    assert list(body["probabilities"]) == info["classes"]
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-5)


def test_parity_20_test_rows(client):
    pipeline = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    split_indices = json.loads((ROOT / "models" / "split_indices.json").read_text(encoding="utf-8"))
    df = load_data()
    columns = get_feature_columns(FEATURE_SET)
    
    test_indices = split_indices["test"][:20]
    for idx in test_indices:
        row = df.loc[idx]
        payload = {}
        for item in features:
            payload[item["id"]] = float(row[item["csv_name"]])
        
        # Test API response
        resp = client.post("/predict", json={"features": payload})
        assert resp.status_code == 200
        body = resp.get_json()
        
        # Test Direct Pipeline Prediction
        direct_frame = pd.DataFrame([{item["csv_name"]: float(row[item["csv_name"]]) for item in features}])
        direct_probs = pipeline.predict_proba(direct_frame)[0]
        classes = list(pipeline.classes_)
        direct_label = classes[int(direct_probs.argmax())]
        
        assert body["label"] == direct_label
        for c, p in zip(classes, direct_probs):
            assert body["probabilities"][c] == pytest.approx(p, abs=1e-5)


def test_realistic_student_scenarios(client):
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    base_values = {}
    for item in features:
        if "options" in item:
            base_values[item["id"]] = item["options"][0]["value"]
        elif "median" in item:
            base_values[item["id"]] = item["median"]
        else:
            base_values[item["id"]] = item["observed_min"]

    # Scenario 1: Model Student (all units approved with high GPA, tuition paid, no debt)
    high_achiever = dict(base_values)
    high_achiever.update({
        "cu_1st_enrolled": 6,
        "cu_1st_evaluations": 6,
        "cu_1st_approved": 6,
        "cu_1st_grade": 16.5,
        "cu_1st_without_evaluations": 0,
        "cu_2nd_enrolled": 6,
        "cu_2nd_evaluations": 6,
        "cu_2nd_approved": 6,
        "cu_2nd_grade": 17.0,
        "cu_2nd_without_evaluations": 0,
        "tuition_fees_up_to_date": 1,
        "debtor": 0,
        "scholarship_holder": 1
    })
    resp_achiever = client.post("/predict", json={"features": high_achiever})
    assert resp_achiever.status_code == 200
    assert resp_achiever.get_json()["label"] == "Graduate"

    # Scenario 2: At-Risk Student (0 approved units, 0 grade, debtor, unpaid tuition)
    at_risk = dict(base_values)
    at_risk.update({
        "cu_1st_enrolled": 6,
        "cu_1st_evaluations": 6,
        "cu_1st_approved": 0,
        "cu_1st_grade": 0.0,
        "cu_1st_without_evaluations": 0,
        "cu_2nd_enrolled": 6,
        "cu_2nd_evaluations": 6,
        "cu_2nd_approved": 0,
        "cu_2nd_grade": 0.0,
        "cu_2nd_without_evaluations": 0,
        "tuition_fees_up_to_date": 0,
        "debtor": 1,
        "scholarship_holder": 0
    })
    resp_at_risk = client.post("/predict", json={"features": at_risk})
    assert resp_at_risk.status_code == 200
    assert resp_at_risk.get_json()["label"] == "Dropout"

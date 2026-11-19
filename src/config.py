from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "data.csv"
MODEL_PATH = ROOT / "models" / "rf_pipeline.joblib"
FEATURES_PATH = ROOT / "models" / "features.json"
MODEL_INFO_PATH = ROOT / "models" / "model_info.json"

SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
FEATURE_SET = "B_full"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

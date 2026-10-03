from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
DATASET_NAME = "bank.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME
MODEL_PATH = MODELS_DIR / "best_model.joblib"
TRAINING_REPORT_PATH = REPORTS_DIR / "metrics.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_metrics.json"
OPTUNA_TRIALS_PATH = TABLES_DIR / "optuna_trials.csv"
TARGET_COLUMN = "y"

DROP_COLUMNS = ("duration",)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
N_TRIALS = 20
PREDICTION_THRESHOLD = 0.5
OPTUNA_STUDY_NAME = "bank_marketing_automl"
MLFLOW_EXPERIMENT_NAME = "bank_marketing_automl"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

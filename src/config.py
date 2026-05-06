from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data-raw"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

TARGET = "success"

NUMERIC_FEATURES = [
    "iyear",
    "imonth",
    "iday",
    "extended",
    "latitude",
    "longitude",
    "specificity",
    "vicinity",
    "doubtterr",
    "multiple",
    "suicide",
    "guncertain1",
    "individual",
    "claimed",
    "INT_LOG",
    "INT_IDEO",
    "INT_MISC",
    "INT_ANY",
]

CATEGORICAL_FEATURES = [
    "region_txt",
    "country_txt",
    "attacktype1_txt",
    "targtype1_txt",
    "targsubtype1_txt",
    "natlty1_txt",
    "weaptype1_txt",
    "weapsubtype1_txt",
]

LEAKAGE_COLUMNS = {
    "nkill",
    "nkillus",
    "nkillter",
    "nwound",
    "nwoundus",
    "nwoundte",
    "property",
    "propextent",
    "propextent_txt",
    "propvalue",
    "ransom",
    "ransomamt",
    "ransompaid",
    "hostkidoutcome",
    "hostkidoutcome_txt",
    "nreleased",
}


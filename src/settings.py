from omegaconf import OmegaConf

PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"

ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

DOWNLOAD_DIRECTORY_MIMICIII = (
    "Path/to/MIMIC-III data"  # Path to the MIMIC-III data.
)


DATA_DIRECTORY_MIMICIII_FULL = OmegaConf.load("configs/data/mimiciii_full.yaml").dir
DATA_DIRECTORY_MIMICIII_50 = OmegaConf.load("configs/data/mimiciii_50.yaml").dir


PROJECT = "<your project name>"
EXPERIMENT_DIR = "files/"  # Path to the experiment directory. Example: ~/experiments
PALETTE = {
    "PLM-ICD": "#E69F00",
}
HUE_ORDER = ["PLM-ICD"]
MODEL_NAMES = {"PLMICD": "PLM-ICD"}
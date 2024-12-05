from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCE = ROOT / "resource"
CONLL_DIR = RESOURCE / "conll"

SEED = 1234
COLUMNS = ["id", "text", "tokens", "token_ids", "label"]

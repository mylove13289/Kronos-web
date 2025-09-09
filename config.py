# config.py
from pathlib import Path

Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH_local": "/Users/longquan/Documents/git/models/models/",
    "MODEL_PATH": "/Users/longquan/Documents/git/py/",
    "SYMBOL": 'x',
    "INTERVAL": 'x',
    "HIST_POINTS": 360,
    "PRED_HORIZON": 24,
    "N_PREDICTIONS": 30,
    "VOL_WINDOW": 24,
}

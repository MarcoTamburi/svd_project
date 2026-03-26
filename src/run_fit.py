# src/run_fit.py
from fit3 import run_fit3

def run(config_path: str, n_components: int):
    if n_components == 3:
        return run_fit3(config_path)
    else:
        raise ValueError("Per ora supporto solo n_components=3 nel test")
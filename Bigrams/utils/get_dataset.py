from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent.parent

def get_dataset():
    names = open(PARENT_DIR / "names.txt").read().splitlines()
    return names
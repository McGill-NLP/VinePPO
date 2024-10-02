from pathlib import Path


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent
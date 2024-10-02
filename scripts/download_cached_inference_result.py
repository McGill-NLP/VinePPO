import json
import sys
import zipfile
from pathlib import Path

import wandb


def get_entity_name() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_account.json"
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    entity = wandb_account["entity"]
    return entity


def get_project_name() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_account.json"
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    project = wandb_account["project"]
    return project


def get_api_key() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_api_key.json"
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    api_key = wandb_account["key"]
    return api_key


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent


def get_wandb_api() -> wandb.Api:
    import os

    os.environ["WANDB_API_KEY"] = get_api_key()
    return wandb.Api(
        overrides={
            "project": get_project_name(),
            "entity": get_entity_name(),
        },
        timeout=120,
        api_key=get_api_key(),
    )


def main():
    # First argument is the wandb run id
    run_id = sys.argv[1]

    download_dir = get_repo_dir() / "cached_inference_results"
    download_dir.mkdir(exist_ok=True, parents=True)

    download_path = download_dir / run_id

    wandb_api = get_wandb_api()

    # Download the file from the run
    wandb_run = wandb_api.run(f"{get_project_name()}/{run_id}")
    files = list(wandb_run.files())
    files = [f for f in files if f.name == "inference_results.zip"]
    assert len(files) == 1

    # Download the file
    files[0].download(download_path, replace=True)

    # Extract the zip file
    print(f"Downloaded inference results to {download_path}, extracting...")
    with zipfile.ZipFile(download_path / "inference_results.zip", "r") as zip_ref:
        zip_ref.extractall(download_path)
    print("Extraction complete!")


if __name__ == "__main__":
    main()

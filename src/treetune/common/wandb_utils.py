import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

import jsonlines
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from treetune.common import nest

CUSTOMIZED_RUN_FRAGMENT = """fragment RunFragment on Run {
    id
    tags
    name
    displayName
    sweepName
    state
    config
    group
    jobType
    commit
    readOnly
    createdAt
    heartbeatAt
    description
    notes
    runInfo {
        program
        args
        os
        python
        gpuCount
        gpu
    }
    host
    systemMetrics
    summaryMetrics
    historyLineCount
    user {
        name
        username
    }
    historyKeys
}"""

import wandb.apis.public as wandb_public_api

wandb_public_api.RUN_FRAGMENT = CUSTOMIZED_RUN_FRAGMENT
from wandb_gql import gql

wandb_public_api.Runs.QUERY = gql(
    """
    query Runs($project: String!, $entity: String!, $cursor: String, $perPage: Int = 50, $order: String, $filters: JSONString) {
        project(name: $project, entityName: $entity) {
            runCount(filters: $filters)
            readOnly
            runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
                edges {
                    node {
                        ...RunFragment
                    }
                    cursor
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
            }
        }
    }
    %s
    """
    % CUSTOMIZED_RUN_FRAGMENT
)
import wandb
from wandb.apis.public import Run


# Taken from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s23.html
def add_sys_path(new_path):
    import sys, os

    # Avoid adding nonexistent paths
    if not os.path.exists(new_path):
        return -1

    # Standardize the path. Windows is case-insensitive, so lowercase
    # for definiteness.
    new_path = os.path.abspath(new_path)
    if sys.platform == "win32":
        new_path = new_path.lower()

    # Check against all currently available paths
    for x in sys.path:
        x = os.path.abspath(x)
        if sys.platform == "win32":
            x = x.lower()
        if new_path in (x, x + os.sep):
            return 0
    sys.path.append(new_path)
    return 1


def get_entity_name() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_account.json"
    # if wandb_account_file.exists():
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    entity = wandb_account["entity"]
    return entity


def get_project_name() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_account.json"
    # if wandb_account_file.exists():
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    project = wandb_account["project"]
    return project


def get_api_key() -> str:
    wandb_account_file = get_repo_dir() / "configs/.wandb_api_key.json"
    # if wandb_account_file.exists():
    with wandb_account_file.open("r") as f:
        wandb_account = json.load(f)
    api_key = wandb_account["key"]
    return api_key


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent


def get_wandb_api() -> wandb.Api:
    import os

    os.environ["WANDB_API_KEY"] = get_api_key()
    return wandb.Api(
        overrides={
            "project": get_project_name(),
            "entity": get_entity_name(),
        },
        timeout=240,
        api_key=get_api_key(),
    )


def get_aggr_dataframe_with_mean_std(
    df: pd.DataFrame, aggr_cols: List[str], metric_name: str
) -> pd.DataFrame:
    xdf = df.groupby(aggr_cols).agg({metric_name: ["mean", "std"]})
    xdf.columns = ["_".join(x) for x in xdf.columns]
    xdf = xdf.reset_index()
    return xdf


def create_mask(df: pd.DataFrame, key_values: Dict[str, str]) -> pd.Series:
    # First, produce mask that is True for all rows based on df
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in key_values.items():
        mask = mask & (df[key] == value)
    return mask


def get_result_name(tags: List[str]) -> str:
    return "_".join(tags)


def get_launcher_id(run: Run) -> str:
    if run.job_type == "agent":
        return run.id

    launcher_tag = [t for t in run.tags if t.startswith("launched_by_")]
    if len(launcher_tag) == 0:
        return None
    launcher_tag = launcher_tag[0]
    return launcher_tag.split("launched_by_")[1]


def load_dataframe_from_jsonlines(path: Path) -> pd.DataFrame:
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame.from_records(data)


def download_and_load_results(
    tags: List[str],
    force_download: bool = False,
    key: str = None,
    return_runs: bool = False,
    save_to_disk: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Run]]]:
    if return_runs:
        assert force_download, "return_runs only works when force_download is True"

    if key is None:
        key = get_result_name(tags)

    result_dir = get_repo_dir() / "results"
    result_dir.mkdir(exist_ok=True, parents=True)

    result_path = result_dir / f"{key}.jsonl"
    if result_path.exists() and not force_download:
        return load_dataframe_from_jsonlines(result_path)

    wandb_api = get_wandb_api()
    runs = wandb_api.runs(
        f"{get_entity_name()}/{get_project_name()}",
        filters={"tags": {"$in": tags}},
    )

    wandb_entity = get_entity_name()
    wandb_project = get_project_name()

    df_data = []
    for run in tqdm(runs, total=len(runs)):
        config = run.config
        config = nest.flatten(config, separator=".")
        config = {f"cfg__{k}": v for k, v in config.items()}

        summary = run.summary._json_dict
        summary = nest.flatten(summary, separator="#")
        summary = {f"sum__{k}": v for k, v in summary.items()}

        run_info = run.run_info
        if run_info is None:
            run_info = {}
        run_info = nest.flatten(run_info, separator=".")
        run_info = {f"runInfo__{k}": v for k, v in run_info.items()}

        group = run.group
        df_data.append(
            {
                "run_group": group,
                "run_name": run.name,
                "job_type": run.job_type,
                "tags": run.tags,
                "launcher_id": get_launcher_id(run),
                "state": run.state,
                "id": run.id,
                "group_url": f"https://wandb.ai/{wandb_entity}/{wandb_project}/groups/{group}",
                "run_url": run.url,
                "created_at": run.created_at,
                "host": run.host,
                **run_info,
                **config,
                **summary,
            }
        )

    print("Building dataframe...")
    df = pd.DataFrame(df_data)

    if save_to_disk:
        print("Saving results to", result_path)
        with jsonlines.open(result_path, mode="w") as writer:
            writer.write_all(df_data)

    if return_runs:
        return df, runs

    return df


def load_inference_result_from_run_id(
    run_id: str,
    file_name: str = "inference_results.zip",
    download_dir: Path = None,
    use_cache: bool = True,
    return_download_path: bool = False,
) -> Dataset:
    assert file_name.endswith(".zip")

    if download_dir is None:
        download_dir = get_repo_dir() / "results" / Path("inference_results")
        download_dir.mkdir(exist_ok=True, parents=True)

    download_path = download_dir / f"{run_id}__{file_name}"
    if use_cache and download_path.exists():
        if return_download_path:
            return Dataset.load_from_disk(str(download_path)), download_path
        else:
            return Dataset.load_from_disk(str(download_path))

    wandb_api = get_wandb_api()

    # Download the file from the run
    wandb_run = wandb_api.run(f"{get_project_name()}/{run_id}")
    files = list(wandb_run.files())
    files = [f for f in files if f.name == file_name]

    # assert with error message
    assert (
        len(files) == 1
    ), f"Expected 1 file with name {file_name}, but found {len(files)} files"

    # Download the file
    files[0].download(download_path, replace=True)

    # Extract the zip file
    import zipfile

    with zipfile.ZipFile(download_path / file_name, "r") as zip_ref:
        zip_ref.extractall(download_path)

    # Load the dataset
    dataset = Dataset.load_from_disk(str(download_path))

    if return_download_path:
        return dataset, download_path
    else:
        return dataset


def download_and_load_file(
    run_id: str,
    file_name: str,
    download_dir: Path = None,
    use_cache: bool = True,
) -> Path:
    if download_dir is None:
        download_dir = get_repo_dir() / "results" / "downloaded_files"
        download_dir.mkdir(exist_ok=True, parents=True)

    download_path = download_dir / run_id
    if use_cache and (download_path / file_name).exists():
        return download_path / file_name

    wandb_api = get_wandb_api()

    # Download the file from the run
    wandb_run = wandb_api.run(f"{get_project_name()}/{run_id}")
    files = list(wandb_run.files())
    files = [f for f in files if f.name == file_name]

    # assert with error message
    assert (
        len(files) == 1
    ), f"Expected 1 file with name {file_name}, but found {len(files)} files"

    # Download the file
    files[0].download(download_path, replace=True)

    return download_path / file_name


def upload_missing_analysis_from_files(run_id: str):
    wandb_api = get_wandb_api()

    def get_analysis_key(file_name):
        file_name = file_name.replace("log_analysis__", "analysis__")
        file_name = file_name.replace(".json", "")
        analysis_name_parts = file_name.split("__")
        prefix = analysis_name_parts[:4]
        suffix = analysis_name_parts[4:]
        prefix = "/".join(prefix)
        suffix = "__".join(suffix)
        return f"{prefix}/{suffix}"

    run = wandb_api.run(f"{get_project_name()}/{run_id}")
    files = list(run.files())
    for file in files:
        file_name = file.name
        if file_name.startswith("log_analysis__"):
            analysis_key = get_analysis_key(file_name)

            # Check if any key with `analysis_key` prefix exists in the run.summary
            if any(
                [sum_key.startswith(analysis_key) for sum_key in run.summary.keys()]
            ):
                continue

            # Download the file and upload it as summary
            # Random temp root to avoid conflicts
            # Create a temporary directory and get its path
            temp_dir = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir.name)

            f = file.download(root=temp_dir_path, replace=True)
            analysis = json.loads(f.read())
            for key, value in analysis.items():
                sum_key = f"{analysis_key}/{key}"
                run.summary[sum_key] = value

            run.save()
            temp_dir.cleanup()


def save_inference_result_to_cloud(
    results: Dataset,
    inference_result_name: str,
    cloud_logger: Optional[wandb.sdk.wandb_run.Run] = None,
    policy="live",
):
    if cloud_logger is None:
        return

    temp_dir_path = Path(tempfile.mkdtemp())
    output_dir = temp_dir_path / inference_result_name

    results.save_to_disk(str(output_dir))

    # Create a zip file of the inference results into output_dir/inference_results.zip
    # This is because the cloud logger only accepts files.
    temp_dir = temp_dir_path / next(tempfile._get_candidate_names())
    temp_dir.mkdir(parents=True, exist_ok=True)

    inference_results_zip = temp_dir / f"{inference_result_name}.zip"
    shutil.make_archive(str(inference_results_zip.with_suffix("")), "zip", output_dir)

    # Then, upload the zip file to the cloud.
    cloud_logger.save(str(inference_results_zip.absolute()), policy=policy)

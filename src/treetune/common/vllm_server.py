import os
import random
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, Callable, Dict

import psutil
import requests

from treetune.common import FromParams
from treetune.common.notebook_utils import get_repo_dir
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


def get_free_port() -> int:
    """Find a free port by binding to port 0 and then releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ensure_executable(script_path: Union[str, Path]):
    """Make sure the server script is executable."""
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, os.stat(script_path).st_mode | 0o111)


def find_and_kill_process(port: int):
    for proc in psutil.process_iter(["pid", "name", "connections"]):
        try:
            connections = proc.info["connections"]
            if connections is None:
                continue

            for conn in connections:
                if conn.laddr.port == port:
                    # If the port matches, print process info and kill the process
                    logger.info(
                        f"Found process {proc.info['name']} with PID {proc.info['pid']} using port {port}"
                    )
                    os.kill(proc.info["pid"], 9)
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def is_port_in_use_error(vllm_log: str) -> bool:
    vllm_log = vllm_log.lower()
    return (
        "error while attempting to bind on address" in vllm_log
        and "address already in use" in vllm_log
    )


def compute_vllm_stats(log_file_path: Path) -> Dict[str, float]:
    gen_throughput_pattern = r"Avg generation throughput: ([\d.]+) tokens/s"
    running_reqs_pattern = r"Running: (\d+) reqs"
    pending_reqs_pattern = r"Pending: (\d+) reqs"
    gpu_cache_pattern = r"GPU KV cache usage: ([\d.]+)%"
    cpu_cache_pattern = r"CPU KV cache usage: ([\d.]+)%"

    total_generation_throughput = 0.0
    total_running_reqs = 0
    total_pending_reqs = 0
    total_gpu_kv_cache_usage = 0.0
    total_cpu_kv_cache_usage = 0.0

    gen_throughput_count = 0
    running_reqs_count = 0
    pending_reqs_count = 0
    gpu_cache_count = 0
    cpu_cache_count = 0

    # Read the log file and process each line
    with open(log_file_path, "r") as file:
        for line in file:
            gen_throughput_match = re.search(gen_throughput_pattern, line)
            if gen_throughput_match:
                total_generation_throughput += float(gen_throughput_match.group(1))
                gen_throughput_count += 1

            running_reqs_match = re.search(running_reqs_pattern, line)
            if running_reqs_match:
                total_running_reqs += int(running_reqs_match.group(1))
                running_reqs_count += 1

            pending_reqs_match = re.search(pending_reqs_pattern, line)
            if pending_reqs_match:
                total_pending_reqs += int(pending_reqs_match.group(1))
                pending_reqs_count += 1

            gpu_cache_match = re.search(gpu_cache_pattern, line)
            if gpu_cache_match:
                total_gpu_kv_cache_usage += float(gpu_cache_match.group(1))
                gpu_cache_count += 1

            cpu_cache_match = re.search(cpu_cache_pattern, line)
            if cpu_cache_match:
                total_cpu_kv_cache_usage += float(cpu_cache_match.group(1))
                cpu_cache_count += 1

    avg_generation_throughput = (
        (total_generation_throughput / gen_throughput_count)
        if gen_throughput_count > 0
        else 0.0
    )
    avg_running_reqs = (
        (total_running_reqs / running_reqs_count) if running_reqs_count > 0 else 0.0
    )
    avg_pending_reqs = (
        (total_pending_reqs / pending_reqs_count) if pending_reqs_count > 0 else 0.0
    )
    avg_gpu_kv_cache_usage = (
        (total_gpu_kv_cache_usage / gpu_cache_count) if gpu_cache_count > 0 else 0.0
    )
    avg_cpu_kv_cache_usage = (
        (total_cpu_kv_cache_usage / cpu_cache_count) if cpu_cache_count > 0 else 0.0
    )

    return {
        "avg_generation_throughput": avg_generation_throughput,
        "avg_running_reqs": avg_running_reqs,
        "avg_pending_reqs": avg_pending_reqs,
        "avg_gpu_kv_cache_usage": avg_gpu_kv_cache_usage,
        "avg_cpu_kv_cache_usage": avg_cpu_kv_cache_usage,
    }


class VLLMServer(FromParams):
    def __init__(
        self,
        seed: int = 42,
        swap_space: int = 16,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 256,
        enable_prefix_caching: bool = False,
        disable_sliding_window: bool = False,
        disable_frontend_multiprocessing: bool = False,
        max_model_len: Optional[int] = None,
        script_path: Optional[Path] = None,
        server_running_check_url: str = "v1/models",
        port: Optional[int] = None,
    ):
        if script_path is None:
            script_path = (
                get_repo_dir() / "scripts" / "start_vllm_server_named_params.sh"
            )
        ensure_executable(script_path)

        assert isinstance(gpu_memory_utilization, float)
        assert 0.0 < gpu_memory_utilization <= 1.0

        self.seed = seed
        self.swap_space = swap_space
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_caching = enable_prefix_caching
        self.disable_sliding_window = disable_sliding_window
        self.disable_frontend_multiprocessing = disable_frontend_multiprocessing
        self.max_model_len = max_model_len
        self.script_path = script_path
        self.port = port
        self.server_running_check_url = server_running_check_url
        self.process: Optional[subprocess.Popen] = None

    def _wait_for_server(
        self,
        launch_func: Callable[[], None],
        timeout: int,
        log_path: Optional[Path] = None,
    ) -> bool:
        """Wait for the server to start by polling the check URL."""
        start_time = time.time()
        while True:
            try:
                response = requests.get(
                    f"http://localhost:{self.port}/{self.server_running_check_url}",
                    proxies={"http": None, "https": None},  # Disabling proxies
                )
                if response.status_code == 200:
                    logger.info("Server is up and responding.")
                    return True
            except requests.ConnectionError:
                # Server is not up yet
                pass
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"An exception occurred while checking the server status: {e}"
                )
                return False

            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for the server to start.")
                return False

            time.sleep(1)

            # Check if the process is still running
            if self.process.poll() is not None:
                logger.error("vLLM process has exited. Restarting...")
                if log_path is not None:
                    with log_path.open("r") as f:
                        vllm_log = f.read()
                        logger.error(f"vLLM Server log:\n{vllm_log}")

                        if is_port_in_use_error(vllm_log):
                            # Get a random number as the port and try again
                            self.port = random.randint(1024, 65533)
                            logger.error(
                                f"Port is already in use. Trying to restart "
                                f"the server using port {self.port}"
                            )

                # Try to restart the server
                launch_func()

    def start_server(
        self,
        hf_ckpt_path_or_model: Union[str, Path],
        log_path: Optional[Path] = None,
        gpu_idx: Optional[int] = None,
        wait_for_response: bool = True,
        timeout: int = 600,
    ) -> str:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Server is already running")

        if self.port is None:
            self.port = get_free_port()

        def launch_func():
            self._launch_process(gpu_idx, hf_ckpt_path_or_model, log_path)

        find_and_kill_process(self.port)
        launch_func()
        logger.info(f"Server started with PID {self.process.pid} on port {self.port}")

        if wait_for_response:
            if not self._wait_for_server(
                launch_func=launch_func, timeout=timeout, log_path=log_path
            ):
                self.stop_server()
                if log_path is not None:
                    with log_path.open("r") as f:
                        logger.error(f"vLLM Server log:\n{f.read()}")
                raise RuntimeError("Server did not start within the expected time.")

        server_url = f"http://localhost:{self.port}/v1"
        return server_url

    def _launch_process(self, gpu_idx, hf_ckpt_path_or_model, log_path):
        # The command arguments:
        command = (
            f"{self.script_path}"
            f" --model {hf_ckpt_path_or_model}"
            f" --port {self.port}"
            f" --seed {self.seed}"
            f" --swap-space {self.swap_space}"
            f" --gpu-memory-utilization {self.gpu_memory_utilization}"
            f" --max-num-seqs {self.max_num_seqs}"
        )
        if gpu_idx is not None:
            command += f" --gpu-idx {gpu_idx}"
        if self.enable_prefix_caching:
            command += " --enable-prefix-caching"
        if self.disable_sliding_window:
            command += " --disable-sliding-window"
        if self.max_model_len is not None:
            command += f" --max-model-len {self.max_model_len}"
        if self.disable_frontend_multiprocessing:
            command += " --disable-frontend-multiprocessing"
        command = shlex.split(command)
        # Redirect both stdout and stderr to the log file if specified
        if log_path is not None:
            with log_path.open("w") as f:
                self.process = subprocess.Popen(command, stdout=f, stderr=f)
        else:
            self.process = subprocess.Popen(command)

    def stop_server(self):
        if self.process is None or self.process.poll() is not None:
            logger.info("Server is not running.")
            return

        self.process.kill()
        time.sleep(3)

        # Use pkill to kill processes matching the pattern
        pattern = f"vllm.entrypoints.openai.api_server.*port {self.port}"
        try:
            subprocess.run(["pkill", "-f", "-9", pattern])
        except subprocess.CalledProcessError as e:
            logger.error(f"An error occurred: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        while True:
            try:
                # Check if any process matches the pattern
                result = subprocess.run(
                    ["pgrep", "-f", pattern], text=True, capture_output=True
                )
                if result.returncode == 0:
                    logger.warning("Process has reappeared!")
                    subprocess.run(["pkill", "-f", "-9", pattern])
                else:
                    break
            except subprocess.CalledProcessError as e:
                logger.error(f"An error occurred while checking the process: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

            time.sleep(1)

        find_and_kill_process(self.port)

        self.process.kill()
        self.process.wait()

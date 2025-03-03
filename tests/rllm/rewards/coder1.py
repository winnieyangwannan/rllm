from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.rule import Rule
import rich
import json 
import sys 

_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}

import os
import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory

import requests

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30

CLI_ARG_SIZE_LIMIT = 1024 * 3

def check_executor_alive(executor):
    try:
        return requests.get(executor + "/").status_code in [200, 404]
    except Exception:
        return False

_ERROR_MSG_PREFIX = "Failed to execute program: "


def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"

    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        f"--timeout=00:00:{timeout}",
    ]

    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command.extend(["python3", "-c", code])
            result = subprocess.run(command,
                                    input=stdin.encode() if stdin else None,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env,
                                    check=False)
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.insert(4, f"--whitelist={tmp.name}")
                command.extend(["python3", tmp.name])
                result = subprocess.run(command,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        env=env,
                                        check=False)

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def minimize_stdio(inputs, outputs):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = [str(x) for x in stdin]
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = [str(x) for x in stdout]
            stdout = "\n".join(stdout)
        # if sys.getsizeof(stdin) > 4 * 1024:
        #     continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin), list(sorted_stdout)


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec_firejail(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def process_fn(idx,example):
    tests = example["tests"]
    stdin_list, stdout_list = minimize_stdio(tests["inputs"], tests["outputs"])

    with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
        futures = []
        for stdin, stdout in zip(stdin_list, stdout_list):
            futures.append(executor.submit(
                remote_check_stdio,
                example["solutions"],
                stdin,
                stdout,
            ))
        for future in as_completed(futures):
            exec_succ, output, stdin, stdout = future.result()
            pass_test = exec_succ and output.strip() == stdout.strip()
            if not pass_test:
                rich.print(f"[bold red]Test code failed for ")
                print(f"*****Failed solution:\n{example['solutions']}\ntask_id:{idx} and stdin_lists:\nstdin_list:{stdin_list}\nstdout_list:{stdout_list}\nstdin:{stdin}\nstdout:{stdout}")
                print(f"{exec_succ = }")
                print(f"{stdin = }", f"{stdout = }")
                if output.startswith(_ERROR_MSG_PREFIX):
                    print("output = \n", output)
                else:
                    print(f"{output = }")
                return False, idx
    return True, idx 

def process_data_with_threads(data, num_threads=256):
    results = []
    false_result = []
    false_id = [] 
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_fn, item["id"], item) for idx, item in enumerate(data)]
        for future in as_completed(futures):
            try:
                result= future.result()
                output = result[0]
                id = result[1]
                if output == False:
                    false_result.append(result)
                    false_id.append(id)
            except Exception as e:
                print(f"Error processing item: {e}")
    print(f"Total test cases: {len(data)} and False test cases: {len(false_result)}")
    #save the false_id
    with open("coder1_false_id_taco.json", "w") as f:
        json.dump(false_id, f)

    return results

if __name__ == "__main__":
    path = "../../../rllm/data/train/code/taco.json"
    with open(path, "r") as f:
        data = json.load(f)
    data = data
    print(f"len(data): {len(data)}")
    results = process_data_with_threads(data)
    print(f"Total test cases: {len(data)}")
from .firejail_exec import code_exec_firejail
from concurrent.futures import ThreadPoolExecutor, as_completed


_ERROR_MSG_PREFIX = "Failed to execute program: "

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


def run_test(in_outs, test=None, debug=False):
    stdin_list, stdout_list = minimize_stdio(in_outs["inputs"], in_outs["outputs"])
    results = []
    with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
        futures = []
        for stdin, stdout in zip(stdin_list, stdout_list):
            futures.append(executor.submit(
                remote_check_stdio,
                test,
                stdin,
                stdout,
            ))
        for future in as_completed(futures):
            exec_succ, output, stdin, stdout = future.result()
            pass_test = exec_succ and output.strip() == stdout.strip()
            if not pass_test:
                print(f"*****Failed solution:\n")
                print(f"{exec_succ = }")
                print(f"{stdin = }", f"{stdout = }")
                if output.startswith(_ERROR_MSG_PREFIX):
                    print("output = \n", output)
                else:
                    print(f"{output = }")
                results.append(False)
            else:
                results.append(True)
    return results


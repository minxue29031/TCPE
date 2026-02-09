import os
import json
import shutil
import subprocess
import traceback
from tqdm.auto import tqdm


COMPILATION_TIMEOUT = 30 # seconds
RUN_TIMEOUT = 15 # seconds

class CompilationError(Exception):
    pass


class TestRuntimeError(Exception):
    pass


def make_accessible(path):
    os.system(f"chmod 0777 {path}")

def build_program(path, bin_path):
    result = subprocess.run(f"dmd -of={bin_path} {path}", shell=True, text=True, capture_output=True, timeout=COMPILATION_TIMEOUT)
    if len(result.stderr) > 0:
        raise CompilationError(result.stderr)


def run_program(path):
    os.system(f"chmod +x {path}")
    result = subprocess.run(path, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    if len(result.stderr) > 0:
        raise TestRuntimeError(result.stderr)
    if "#Results:" not in result.stdout:
        raise TestRuntimeError(f"Result does not conform to pattern: '{result.stdout}'")
    correct, total = result.stdout.split("#Results:")[-1].split(", ")
    correct = int(correct)
    total = int(total)
    return correct, total


def test(path, run_dir):
    file = os.path.split(path)[-1]
    source_code_file = os.path.join(run_dir, file)
    bin_file = os.path.join(run_dir, file.split(".")[-1])
    shutil.copyfile(path, source_code_file)
    build_program(source_code_file, bin_file)
    return run_program(bin_file)


def main():
    paths_to_test = os.listdir("code_under_test")
    run_dir = "run_dir"
    results = []
    for path in tqdm(paths_to_test):
        full_path = os.path.join("code_under_test", path)
        os.makedirs(run_dir, exist_ok=True)
        partial_result = {
            "path": path,
        }
        try:
            correct, total = test(full_path, run_dir)
        except Exception as e:
            error_type = "Exception"
            if isinstance(e, CompilationError):
                error_type = "Compilation"
            elif isinstance(e, TestRuntimeError):
                error_type = "RuntimeError"
            elif isinstance(e, subprocess.TimeoutExpired):
                error_type = "TimeoutExpired"
            else:
                traceback.print_exc()
            results.append({
                "description": repr(e),
                "error": error_type,
                "success": False,
                **partial_result
            })
            continue
        results.append({
            "correct": correct,
            "total": total,
            "success": correct == total,
            "error": "Correctness" if correct != total else "No",
            **partial_result
        })
        shutil.rmtree(run_dir)
    with open("results.json", "w", encoding="utf8") as f:
        json.dump(results, f)
    make_accessible("results.json")


if __name__ == "__main__":
    main()
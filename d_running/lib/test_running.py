import os
import json
import shutil

from pathlib import Path
from typing import List, Dict

IMAGE = "alpine"
CONTAINER_NAME = "d_tests"
DEPENDENCIES = ["gcc", "g++", "dmd", "python3", "py3-tqdm"]
PATH_TO_SCRIPTS = Path(__file__).absolute().parent.joinpath("scripts")


def get_inner_cmd():
    cmd = f"apk add -q {' '.join(DEPENDENCIES)}"
    cmd += " && python3 inside_container_main.py"
    return cmd


def get_docker_cmd(path_to_workspace: Path):
    cmd = "docker run -it --rm "
    cmd += f"--name {CONTAINER_NAME} "
    cmd += f"-v {path_to_workspace}:/workspace "
    cmd += f"-w /workspace "
    # IMAGE
    cmd += f"{IMAGE} "
    # command
    cmd += f'sh -c "{get_inner_cmd()}"'
    return cmd


def workspace_setup(path_to_workspace: Path, file_contents: Dict[str, str]):
    # clear workspace of old junk
    if os.path.isdir(path_to_workspace):
        shutil.rmtree(path_to_workspace)
    os.makedirs(path_to_workspace)
    
    # copy main script
    shutil.copyfile(PATH_TO_SCRIPTS.joinpath("inside_container_main.py"), path_to_workspace.joinpath("inside_container_main.py"))

    # paths to test
    code_under_test_path = path_to_workspace.joinpath("code_under_test")
    os.makedirs(code_under_test_path)
    for filename, file_content in file_contents.items():
        with open(code_under_test_path.joinpath(filename), "w", encoding="utf8") as f:
            f.write(file_content)


def test_paths(paths_to_test: List, path_to_workspace: Path):
    file_contents = {}
    for path in paths_to_test:
        _, filename = os.path.split(path)
        with open(path, "r", encoding="utf8") as f:
            file_contents[filename] = f.read()
    
    return run_tests(file_contents, path_to_workspace)


def run_tests(file_contents: Dict[str, str], path_to_workspace: Path):
    workspace_setup(path_to_workspace, file_contents)
    os.system(get_docker_cmd(path_to_workspace))
    with open(path_to_workspace.joinpath("results.json"), "r", encoding="utf8") as f:
        results = json.load(f)
    return results

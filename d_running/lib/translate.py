import os
import re
import sys
import time
import subprocess

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent.parent

sys.path.insert(0, str(ROOT_DIR))

from d_running.lang_processors.java_processor import JavaProcessor
from d_running.lang_processors.d_processor import DProcessor

from d_running.lib import REPLACEMENT_MARKER

java_proc = JavaProcessor()
d_proc = DProcessor()


def get_prompt(tok: AutoTokenizer, to_translate: str) -> str:
    return tok.apply_chat_template(
        [
            dict(
                role="user",
                content=f"""Translate the following Java function to its equivalent function in the D programming language (dlang). Only provide the completed function.
```{to_translate}```""",
            ),
            dict(role="assistant", content="""```d"""),
        ],
        tokenize=False,
    )[: -len(" </s>")]


def get_gold_function(code, lang="java"):
    # parsing using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    to_fill = code.find(REPLACEMENT_MARKER[lang])
    function = code[function_start : to_fill - 1]

    return function


def translate(model, tok, to_translate, max_new_tokens=400):
    device = next(model.parameters()).device
    prompt = get_prompt(tok, to_translate)
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        add_special_tokens=False,
    ).to(device)
    out = model.generate(
        **inputs,
        do_sample=False,
        num_return_sequences=1,
        max_new_tokens=max_new_tokens,
    )

    return tok.batch_decode(out)[0]


def preprocess_gold_function(code):
    code = code.strip()

    # remove qualifiers from return type
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name) + 1
    return_type = code[function_start:function_name].strip()
    code = code.replace(return_type, return_type.split(" ")[-1], 1)

    code = code.replace("f_gold", "solution")  # more natural function name

    code = _format_java(code)

    return code


def _format_java(code):
    code = java_proc.detokenize_code(code)
    code = (
        code.replace(" ( ", "(")
        .replace(" )", ")")
        .replace(" . ", ".")
        .replace(" ++", "++")
        .replace("++ ", "++")
        .replace(" [", "[")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace(" ,", ",")
    )
    code = re.sub(r"' ([^'])", r"'\1", code)
    code = re.sub(r"([^']) '", r"\1'", code)

    code = re.sub(r"([^ ])\(", r"\1 (", code)

    # manually handle semicolons to handle for loops
    in_parens = 0
    new_code = ""
    code = code.replace(";\n", ";")
    for c in code:
        if "(" == c:
            in_parens += 1
        if ")" == c:
            in_parens -= 1
        if c != ";":
            new_code += c
            continue
        if in_parens > 0:
            new_code += c
            continue
        new_code += c + "\n"
    code = new_code.replace(" ;", ";")

    code = code.replace(" { ", " {\n").replace(" }", "\n}\n")

    new_code = ""
    indentation = 0
    for line in code.split("\n"):
        line = line.strip()
        if line == "":
            continue

        indentation -= line.count("}")
        new_code += indentation * 4 * " " + line + "\n"
        indentation += line.count("{")
    code = new_code

    code = code.strip()
    return code


def extract_answer_function(response):
    answer_start = response.find("[/INST]")
    answer = response[answer_start:]

    code_start = answer.find("```d") + len("```d")
    code_end = len(answer)  # assume no end of code by default
    if answer.count("```") > 1:
        code_end = answer.find("```", code_start)
    code = answer[code_start:code_end]

    functions, class_functions = d_proc.extract_functions(code, tokenized=False)
    if len(functions) > 0:
        return functions[0]
    if len(class_functions) > 0:
        return class_functions[0]
    return code


def translate_function(model, tok, function_source):
    translated = translate(model, tok, function_source)
    d_function = extract_answer_function(translated)

    return d_function


def generate_translations_multi_processed(
    model_path, d_params_path, originals_path, outs_path
):
    total_devices = torch.cuda.device_count()
    device_list = list(range(total_devices))
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_list = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    processes = []
    script_path = ROOT_DIR / "d_running" / "lib" / "scripts" / "generate_part_of_translations.py"

    for i in range(total_devices):
        specific_env = os.environ.copy()
        specific_env["CUDA_VISIBLE_DEVICES"] = str(device_list[i])
        cmd = [
            sys.executable,
            str(script_path),
            "--index",
            str(i),
            "--total_count",
            str(total_devices),
            "--model_path",
            model_path,
            "--d_params_path",
            d_params_path,
            "--originals_path",
            originals_path,
            "--outs_path",
            outs_path,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=specific_env,
        )
        processes.append(proc)

    number_of_files = len(os.listdir(d_params_path))

    pbar = tqdm(total=number_of_files)

    while len(processes) > 0:
        running_processes = []
        for proc in processes:
            if proc.poll() is None:
                running_processes.append(proc)
        processes = running_processes

        new_count = len(os.listdir(outs_path)) - pbar.n
        if new_count > 0:
            pbar.update(new_count)
        time.sleep(0.15)

    pbar.close()


def generate_translations_multi_processed_TC(
    model_path, tc_layer_path, d_params_path, originals_path, outs_path
):
    total_devices = torch.cuda.device_count()
    device_list = list(range(total_devices))
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_list = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    processes = []
    script_path_tc = ROOT_DIR / "d_running" / "lib" / "scripts" / "generate_part_of_translations_TC.py"

    for i in range(total_devices):
        specific_env = os.environ.copy()
        specific_env["CUDA_VISIBLE_DEVICES"] = str(device_list[i])
        cmd = [
            sys.executable,
            str(script_path_tc),
            "--index",
            str(i),
            "--total_count",
            str(total_devices),
            "--model_path",
            model_path,
            "--d_params_path",
            d_params_path,
            "--originals_path",
            originals_path,
            "--outs_path",
            outs_path,
            "--tc_layer_path",
            tc_layer_path,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=specific_env,
        )
        processes.append(proc)

    number_of_files = len(os.listdir(d_params_path))

    pbar = tqdm(total=number_of_files)

    while len(processes) > 0:
        running_processes = []
        for proc in processes:
            if proc.poll() is None:
                running_processes.append(proc)
        processes = running_processes

        new_count = len(os.listdir(outs_path)) - pbar.n
        if new_count > 0:
            pbar.update(new_count)
        time.sleep(0.15)

    pbar.close()

import os
import sys
from pathlib import Path
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from d_running.lib.models import load_model_and_tokenizer_with_TC_layer
from d_running.lib.translate import (
    get_gold_function,
    preprocess_gold_function,
    translate_function,
)
from d_running.lib.translate import generate_translations_multi_processed_TC, extract_answer_function, get_prompt


def get_files_to_translate(d_params_path, outs_path):
    files = os.listdir(d_params_path)
    already_existing_translations = os.listdir(outs_path)
    files = [f for f in files if f not in already_existing_translations]
    return files

MULTI_GPU_TRANSLATION_ENABLED = False

def generate_translations(
    model, tok, d_params_path, originals_path, results_path=None, translation_batch_size=14):
    outs_path = results_path
    os.makedirs(outs_path, exist_ok=True)

    # Get time to measure the performance
    start_time = time.time()
    files = get_files_to_translate(d_params_path, outs_path)

    # Get all files from the originals_path, preprocess them and store them in a list
    codes_to_translate = []
    for file in files:
        original_path = os.path.join(originals_path, file.replace(".d", ".java"))
        with open(original_path, "r", encoding="utf8") as f:
            code = f.read()
            gold_function = get_gold_function(code)
            gold_function = preprocess_gold_function(gold_function)
        codes_to_translate.append(gold_function)

    prompts_all = [get_prompt(tok, code ) for code in codes_to_translate]

    all_results = []
    for i in range(0, len(prompts_all), translation_batch_size):
        print("->translate samples batch", i)
        batch_results = translate_batch(model, prompts_all[i:i + translation_batch_size], tok)
        all_results.extend(batch_results)

    d_functions = [extract_answer_function(result ).replace("solution", "f_filled")
                   for result in all_results]

    for i, file in enumerate(files):
        try:
            out_path = os.path.join(outs_path, file)
            with open(out_path, "w", encoding="utf8") as f:
                f.write(d_functions[i])
        except Exception as e:
            print(f"Saving translated function {file} failed with {e}")

    end_time = time.time()


def translate_batch(model, prompts_all, tok):
    """Tokenize, translate, and decode a batch of prompts"""

    tok.pad_token = tok.eos_token 
    device = next(model.parameters()).device
    input_batch = tok(
        prompts_all,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    max_new_tokens = 400   
    out = model.generate(
        **input_batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    all_results = tok.batch_decode(out)
    return all_results

import os
import sys
from pathlib import Path
import time

# Try to avoid memory fragmentation on GPUs
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



# todo: move this to the config
MULTI_GPU_TRANSLATION_ENABLED = False


def generate_translations(
    model, tok, d_params_path, originals_path, results_path=None, translation_batch_size=14):
    """
    Generate translations for all files in d_params_path and save them to results_path
    Processing is done in batches on a single GPU
    (and so far no batching on multiple GPUs)
    batch_size: number of files to translate in one batch; todo: move to configs
    """

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

    # Turn sources into prompts
    prompts_all = [get_prompt(tok, code ) for code in codes_to_translate]

    ###########################
    # The core of this function
    ###########################
    all_results = []
    for i in range(0, len(prompts_all), translation_batch_size):
        print("------->translate samples batch", i)
        batch_results = translate_batch(model, prompts_all[i:i + translation_batch_size], tok)
        all_results.extend(batch_results)

    # Postprocessing and saving the results
    # Extract the answer functions from all the translated code
    d_functions = [extract_answer_function(result ).replace("solution", "f_filled")
                   for result in all_results]

    # Save all the translated functions to the output path
    for i, file in enumerate(files):
        try:
            out_path = os.path.join(outs_path, file)
            with open(out_path, "w", encoding="utf8") as f:
                f.write(d_functions[i])
        except Exception as e:
            print(f"Saving translated function {file} failed with {e}")

    # Measure the performance
    end_time = time.time()


def translate_batch(model, prompts_all, tok):
    """Tokenize, translate, and decode a batch of prompts"""

    # Tokenize all prompts using left padding
    # Read this: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    tok.pad_token = tok.eos_token  # Most LLMs don't have a pad token by default
    device = next(model.parameters()).device
    input_batch = tok(
        prompts_all,
        return_tensors="pt",
        padding=True,
        # padding=False,    # original code
        # truncation=True,    # created warning that no maximum length is provided
        add_special_tokens=False,
    ).to(device)

    # src code as doc: https://github.com/huggingface/transformers/blob/dd16acb8a3e93b643aa374c9fb80749f5235c1a6/src/transformers/generation/utils.py#L1879
    max_new_tokens = 400    # todo: move to configs
    out = model.generate(
        **input_batch,
        do_sample=False,
        # num_return_sequences=1,   # Not sure what it does but doesn't sound good... I just remove it ;-)
        max_new_tokens=max_new_tokens,
    )
    # Debug only
    # result_one_sample = tok.batch_decode(out)[0]
    # logger.info(f"translated one sample with {len(result_one_sample)} tokens")
    all_results = tok.batch_decode(out)
    return all_results

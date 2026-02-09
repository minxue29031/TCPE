import os
import sys
import json
import shutil
from pathlib import Path
import torch
import argparse
from tqdm import tqdm
import wandb
sys.path.append(os.path.join(os.path.dirname(__file__), "d_running"))
from lib.test_running import run_tests
from lib.models import load_model_and_tokenizer
from lib.translate import get_gold_function, preprocess_gold_function, translate_function, REPLACEMENT_MARKER
from lib.translate import generate_translations_multi_processed


def test_translations(reslut_path, d_params_path, container_workspace):
    results_path = os.path.join(reslut_path, "results.json")
    file_contents = {}

    for file in os.listdir(d_params_path):
        try:
            d_path = os.path.join(d_params_path, file)
            function_path = os.path.join(reslut_path, file)
            if not os.path.isfile(d_path):
                continue
            if not os.path.isfile(function_path):
                continue
            with open(function_path, "r", encoding="utf8") as f:
                d_function = f.read()

            with open(d_path, "r", encoding="utf8") as f:
                code = f.read()
            code = code.replace(REPLACEMENT_MARKER["d"], d_function)

            file_contents[file] = code
        except Exception as e:
            print(f"{file} failed with {e}")


    current_dir = Path(__file__).parent
    custom_workspace_path = current_dir / container_workspace
    
    results = run_tests(file_contents, custom_workspace_path)
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results, f)
    
    return results


def unit_test(reslut_path, d_params_path, originals_path, translate_only, invalidate_cache, container_workspace):
    if invalidate_cache:
        shutil.rmtree(reslut_path , ignore_errors=True)
    if translate_only:
        return
    results = test_translations(reslut_path, d_params_path, container_workspace)
    errors = [result for result in results if not result["success"]]
    success_rate = 1 - len(errors)/len(results)

    print(f"Tested {len(results)} cases, had {len(errors)} errors. {success_rate*100:.2f}% success rate")
    return success_rate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reslut_path")
    parser.add_argument("--d_params_path", default="./d_running/data/d_with_params")
    parser.add_argument("--originals_path", default="./d_running/geeks4geeks/")
    parser.add_argument("--translate_only", "-t", action="store_true")
    parser.add_argument("--invalidate_cache", "-i", action="store_true")
    parser.add_argument("--container_workspace", default="container_workspace", help="Path to the container workspace")
    args = parser.parse_args()  

    unit_test(args.reslut_path, args.d_params_path, args.originals_path, args.translate_only, args.invalidate_cache, args.container_workspace)

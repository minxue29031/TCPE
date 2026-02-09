import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from tcpe.tcpe_pre import apply_rome_tc_to_model
from tcpe.tcpe_main import apply_tcpe_to_model
from util import nethook
from util.generate import generate_fast
from util.globals import *
from tqdm import tqdm
import shutil  
import sys

sys.path.append("./d_running")
import test_model_TC as translating_TC
from lib.test_running import run_tests
from lib.models import load_model_and_tokenizer
from lib.translate import get_gold_function, preprocess_gold_function, translate_function, REPLACEMENT_MARKER
from lib.translate import generate_translations_multi_processed


def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    output_folder_path: str,
    d_params_path: str,
    originals_path:str,
    translation_batch_size: int,
    upd_decay: float,
    lim: float,
    save_path: Optional[str] = None,
    save_k_v_upd: bool= False,
    abla_exp: bool= False,
    alg_name: str = "TCPE_pre",
    hparams=None,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:

    nethook.set_requires_grad(True, model)
    os.makedirs(output_folder_path, exist_ok=True)
    
    print("Hyperparameters:", hparams)
    print(f"Applying model editing using the {alg_name} algorithm.")

 
    if alg_name == "TCPE_pre":
        # Apply TCPE_pre editing and save original weights
        print_loud("Applying TCPE_pre model editing...")
        model_new, orig_weights = apply_rome_tc_to_model(
            model, tok, requests, hparams, save_k_v_upd, save_path, return_orig_weights=True
        )
        
        print_loud("Generating translations using Codellama & TCPE_pre...")
        translating_TC.generate_translations(
            model_new, tok, d_params_path, originals_path, output_folder_path, translation_batch_size
        )
        
        return model_new, orig_weights

    elif alg_name == "TCPE":
        # Apply TCPE editing and save original weights
        print_loud("Applying TCPE model editing...")
        model_new, orig_weights = apply_tcpe_to_model(
            model, tok, requests, hparams, abla_exp, save_path, upd_decay, lim, return_orig_weights=True
        )
        
        print_loud("Generating translations using Codellama & TCPE ...")
        translating_TC.generate_translations(
            model_new, tok, d_params_path, originals_path, output_folder_path, translation_batch_size
        )
        return model_new, orig_weights

    elif alg_name == "noKE":
        # Perform translation without any model editing (pure Codellama)
        print_loud("Generating translations using pure Codellama...")
        translating_TC.generate_translations(
            model, tok, d_params_path, originals_path, output_folder_path, translation_batch_size
        )
        return model, {}

    else:
        # Raise an error if the algorithm name is unknown
        raise ValueError(f"Unknown algorithm name: {alg_name}")
        
        
        
def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying TCPE to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution

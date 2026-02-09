from copy import deepcopy
from typing import Dict, List, Tuple
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from util import nethook
from util.generate import generate_fast
import numpy as np
import wandb
from .compute_u import compute_u
from .compute_v import compute_v
from .tcpe_hparams import TCPEHyperParams

CONTEXT_TEMPLATES_CACHE = None

    
def apply_tcpe_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: TCPEHyperParams,
        abla_exp,
        save_path: str, 
        upd_decay: float,
        lim: float,
        copy=False,
        return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    if copy:
        model = deepcopy(model)

    weights_copy = {}
    
    index_file_path = os.path.join(save_path, "index.pt")
    value_file_path = os.path.join(save_path, "value.pt")
    upd_file_path = os.path.join(save_path, "upd.pt")
    
    # Load active neruon to update_matrix
    if lim >= 0:
        print(">> Load k_star related update_matrix")
        update_matrix, kstar_num = load_active_neurons(abla_exp, index_file_path, value_file_path, upd_file_path, lim)
        wandb.log({"kstar_num": kstar_num})   
        
    elif lim == -1:
        print(">> Load complete update_matrix")
        with open(upd_file_path, "rb") as f:
            update_matrix = pickle.load(f)

    with torch.no_grad():
        # Get the parameter to update
        w_name = f"model.layers.{hparams.layers[0]}.mlp.down_proj.weight"
        w = nethook.get_parameter(model, w_name)

        if return_orig_weights and w_name not in weights_copy:
            weights_copy[w_name] = w.detach().clone()

        w[...] += torch.tensor(update_matrix).to(w.device) / upd_decay
        
    return model, weights_copy



def load_active_neurons(abla_exp: bool, index_file_path: str, value_file_path: str, upd_file_path: str, lim: float) -> torch.Tensor:
    VALUE = torch.load(value_file_path)
    INDEX = torch.load(index_file_path)
    upd_tensor = torch.load(upd_file_path)
    
    mask = VALUE >= lim
    filtered_index = INDEX[mask]
    kstar_num = len(filtered_index)
    
    mask = torch.zeros(upd_tensor.shape[1], dtype=torch.bool)
    for col in filtered_index:
        if 0 <= col < upd_tensor.shape[1]:
            mask[col] = True

    if abla_exp:
        print("<< Ablation Experiment >>")
        upd_tensor[:, mask] = 0
        
    else:
        print("<< TCPE Knowledge Editing Method >>")
        upd_tensor[:, ~mask] = 0  

    return upd_tensor, kstar_num
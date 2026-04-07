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

    
def apply_tcpe(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: TCPEHyperParams,
    abla_exp,
    save_path_list: List[str],
    upd_decay_list: List[float],
    lim: List[float],  
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:

    """
    Logs total kstar count, intersection and union of INDEX sets.
    """
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    index_sets = []
    total_kstar_num = 0

    with torch.no_grad():
        w_name = f"model.layers.{hparams.layers[0]}.mlp.down_proj.weight"
        w = nethook.get_parameter(model, w_name)

        if return_orig_weights and w_name not in weights_copy:
            weights_copy[w_name] = w.detach().clone()
 
        for save_path, upd_decay, lim_i in zip(save_path_list, upd_decay_list, lim):
            index_file_path = os.path.join(save_path, "index.pt")
            value_file_path = os.path.join(save_path, "value.pt")
            upd_file_path = os.path.join(save_path, "upd.pt")

            if lim_i >= 0:
                print(f"===== Load k_star update_matrix from {save_path} =====")
                update_matrix, kstar_num, index_tensor = load_update_matrix(
                    abla_exp, index_file_path, value_file_path, upd_file_path, lim_i
                )
                index_sets.append(set(index_tensor.tolist()))
                total_kstar_num += kstar_num
                wandb.log({f"kstar_num_{os.path.basename(save_path)}": kstar_num})
                
                
            elif lim_i == -1:
                print(f"===== Load complete update_matrix from {save_path} =====")
                with open(upd_file_path, "rb") as f:
                    update_matrix = torch.load(f, weights_only=False)

            w[...] += torch.tensor(update_matrix).to(w.device) / upd_decay
 
                    

    # Compute union and intersection of INDEX sets
    if index_sets:
        index_union = set.union(*index_sets)
        index_intersection = set.intersection(*index_sets) if len(index_sets) > 1 else index_sets[0]
        print(f"[k* Summary] Total k*: {total_kstar_num}")
        print(f"[k* Summary] INDEX union length: {len(index_union)}")
        print(f"[k* Summary] INDEX intersection length: {len(index_intersection)}")
        wandb.log({
            "total_kstar_num": total_kstar_num,
            "index_union_len": len(index_union),
            "index_intersection_len": len(index_intersection),
        })

    return model, weights_copy




def load_update_matrix(
    abla_exp: bool,
    index_file_path: str,
    value_file_path: str,
    upd_file_path: str,
    lim: float
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    # Read k_star VALUE and INDEX from the files, load upd_matrix
    VALUE = torch.load(value_file_path)
    INDEX = torch.load(index_file_path)
    upd_tensor = torch.load(upd_file_path)

    # Filter INDEX based on VALUE
    mask = VALUE >= lim
    filtered_index = INDEX[mask]
    kstar_num = len(filtered_index)

    # Create a mask for the valid columns
    mask_cols = torch.zeros(upd_tensor.shape[1], dtype=torch.bool)
    for col in filtered_index:
        if 0 <= col < upd_tensor.shape[1]:
            mask_cols[col] = True

    if abla_exp:
        print("<<<<<< Ablation Experiment >>>>>>")
        upd_tensor[:, mask_cols] = 0
        non_zero_cols = torch.nonzero(torch.tensor(upd_tensor.sum(0) != 0)).squeeze(1)
        print(f"Shape of remaining non-zero columns after ablation: {non_zero_cols.shape}")
        
    else:
        print("<<<<<< Knowledge Editing Based on k_star >>>>>>")
        upd_tensor[:, ~mask_cols] = 0
        non_zero_cols = torch.nonzero(torch.tensor(upd_tensor.sum(0) != 0)).squeeze(1)
        print(f"Shape of remaining non-zero columns after KE: {non_zero_cols.shape}")

    return upd_tensor, kstar_num, filtered_index

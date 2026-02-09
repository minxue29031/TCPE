from copy import deepcopy
from typing import Dict, List, Tuple
import re
import torch
import pickle
import uuid
import os
from typing import Optional
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .tcpe_hparams import TCPEHyperParams

CONTEXT_TEMPLATES_CACHE = None
 

def apply_rome_tc_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: TCPEHyperParams,
        save_k_v_upd,
        save_path: Optional[str] = None,
        copy=False,
        return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    if copy:
        model = deepcopy(model)

    weights_copy = {}
    
    for i, request in enumerate(requests):
        deltas, non_zero_k_star_count, non_zero_k_star_indices, non_zero_k_star_values = execute_rome_tc(model, tok, request, hparams)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                c1=upd_matrix.norm()
                w = nethook.get_parameter(model, w_name)
                
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()
                    
                c2=w.norm()
                w[...] += upd_matrix 

                wandb.log({
                    "upd_matrix": c1,
                    "ori_matrix": c2,
                    "orimatrix_divide_updmatrix": c2/c1,
                    "non_zero_k_star_count": non_zero_k_star_count
                })     

                if save_k_v_upd:
                    os.makedirs(save_path, exist_ok=True) 
                    
                    index_file_path = os.path.join(save_path, "index.pt")
                    torch.save(non_zero_k_star_indices, index_file_path)
                    
                    value_file_path = os.path.join(save_path, "value.pt")
                    torch.save(non_zero_k_star_values, value_file_path)
                 
                    upd_file_path = os.path.join(save_path, "upd.pt")
                    upd_matrix_num = upd_matrix.cpu().numpy()
                    torch.save(upd_matrix_num, upd_file_path)
                    

    return model, weights_copy 


    
def execute_rome_tc(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: TCPEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:


    request = deepcopy(request)
    print(
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    deltas = {}
    for layer in sorted(hparams.layers):
        left_vector, k_star = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        
        non_zero_k_star = (k_star != 0)
        non_zero_k_star_count = non_zero_k_star.sum().item() 
        non_zero_k_star_indices = non_zero_k_star.nonzero(as_tuple=True)[0]
        non_zero_k_star_values = k_star[non_zero_k_star]
        
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            k_star,
            get_context_templates(model, tok, hparams.context_template_length_params),
        ).type(left_vector.dtype)
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
            
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas, non_zero_k_star_count, non_zero_k_star_indices, non_zero_k_star_values


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError("Check for bugs in the code?")


def clear_context_templates():
    global CONTEXT_TEMPLATES_CACHE
    CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        prompt = tok.bos_token
        if tok.chat_template is not None:
            pass
            
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "{{").replace("}", "}}") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        prompt,
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
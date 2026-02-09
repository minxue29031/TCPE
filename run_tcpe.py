import os
import torch
import gc
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface.transcoder_adapter import TranscoderAdapter
from tcpe.tcpe_hparams import TCPEHyperParams
from experiments.py.demo import demo_model_editing
from d_running.unit_test import unit_test
from d_running.evaluation import run_eval, create_clustering
import wandb

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run model editing with TCPE.")
    
    # Model loading and KE types
    parser.add_argument('--MODEL_NAME', type=str, required=True, help='Accepts either a Hugging Face model identifier')
    parser.add_argument("--layer_num", type=int, required=True, help="Layer number for model editing")
    parser.add_argument('--TC_layer_file', type=str, required=False, help='Path to TransCoder layer file')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to save translation results')
    parser.add_argument('--alg_name', type=str, required=True, help='TCPE_pre, TCPE, noKE')

    # Arguments for KE
    parser.add_argument('--mlp_or_tc', type=str, required=False, help='Load covariance matrix: MLP, TC_ef2, TC_ef3, TC_ef4, TC_ef6, TC_ef8, TC_ef10')
    parser.add_argument("--mom2_n_samples", type=int, required=False, help="Number of samples for computing covariance matrix")
    parser.add_argument("--mom2_dataset", type=str, required=False, help="dataset for computing covariance matrix")
    parser.add_argument("--epsilon", type=float, default=0, required=False, help="Avoid covariance matrix diagonal being 0: 1e-8 or 0")
    parser.add_argument("--v_num_grad_steps", type=int, required=False, help="Number of gradient steps for optimization")
    parser.add_argument("--v_lr", type=float, required=False, help="Learning rate for optimization")
    parser.add_argument("--clamp_norm_factor", type=int, required=False, help="Clamp factor for weight updates")
    parser.add_argument('--edit_sequence_path', type=str, required=False, help='KE requests & target cluster')
    parser.add_argument('--i', type=int, required=True, help='Index for selecting a specific sequence')
    parser.add_argument('--fact_token', type=str, default="subject_last",  required=False, help='subject_last, last')

    # Arguments for KE_STAR
    parser.add_argument("--save_k_v_upd", action="store_true", help="Save the update matrix to a file")
    parser.add_argument('--save_path', type=str, required=False, default="./k_v_upd/", help='Path to save/load key,value,upd_matrix file')
    parser.add_argument("--upd_decay", type=float, required=False, default=1.0 ,help="Decay factor for weight updates")
    parser.add_argument("--lim", type=float, required=False, help="Limit to select k_star active points with varying activity levels")
    parser.add_argument("--abla_exp", action="store_true", help="Perform ablation experiment")

    # Arguments for unit test & evaluation
    parser.add_argument('--container_workspace', type=str, required=True, default="container_workspace")
    parser.add_argument("--d_params_path", type=str, required=False, default="d_running/data/d_with_params")
    parser.add_argument("--originals_path", required=False, default="d_running/geeks4geeks/")
    parser.add_argument("--translate_only", "-t", action="store_true")
    parser.add_argument("--invalidate_cache", "-i", action="store_true")
    parser.add_argument("--translation_batch_size", type=int, required=True, help="Translation batch size")
    parser.add_argument('--preKE_cluster_result', type=str, required=True, help='Path to cluster without KE')
    parser.add_argument("--max_length", type=int, required=False, default=8, help="classify error based on first n tokens.")
    parser.add_argument("--threshold", type=float, required=False, default=0.9, help="classify error based on similarity.")
    parser.add_argument("--embedding_model", type=str, required=False, default="Alibaba-NLP/gte-base-en-v1.5")


    return parser.parse_args()

# Initialize WandB
def init_wandb(args):
    wandb.init(
        project="project_name",
        config={
            "TC_layer_file": args.TC_layer_file,
            "KE_or_noKE": args.alg_name,
            "request_path": args.edit_sequence_path,
            "lim": args.lim,
            "upd_decay": args.upd_decay,
            "abla": args.abla_exp
        }
    )

# Load the model and tokenizer
def load_model_and_tokenizer(args):
    print(f">> Loading Pure model from {args.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    
    if args.TC_layer_file:
        print(f">> Loading Transcoder from {args.TC_layer_file}...")
        adapter = TranscoderAdapter.load(args.TC_layer_file)
        device = next(model.base_model.layers[args.layer_num].mlp.parameters()).device
        model.base_model.layers[args.layer_num].mlp = adapter.to(model.dtype).to(device)
    
    tok = AutoTokenizer.from_pretrained(args.MODEL_NAME)
    
    return model, tok

 
def set_hyperparameters(args):
    hparams = TCPEHyperParams(**{
        "layers": [args.layer_num],
        "fact_token": args.fact_token,
        "v_num_grad_steps": args.v_num_grad_steps,
        "v_lr": args.v_lr,
        "v_loss_layer": 31,
        "v_weight_decay": 0.001,
        "clamp_norm_factor": args.clamp_norm_factor,
        "kl_factor": 0.0625,
        "mom2_adjustment": True,
        "context_template_length_params": [[10, 1]], 
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn.o_proj",
        "ln_f_module": "model.norm",
        "lm_head_module": "lm_head",
        "mom2_dataset": args.mom2_dataset,
        "mom2_n_samples": args.mom2_n_samples,
        "mom2_dtype": "float32",
        "mlp_or_tc": args.mlp_or_tc,
        "epsilon": args.epsilon
    })
    
    wandb.config.update(hparams.__dict__)
    
    return hparams

# Apply model editing
def apply_model_editing(model, tok, request, args, hparams):
    demo_model_editing(
        model, tok, request,  
        args.output_file_path, 
        args.d_params_path,
        args.originals_path, 
        args.translation_batch_size,
        args.upd_decay, 
        args.lim,
        args.save_path, 
        args.save_k_v_upd,  
        args.abla_exp, 
        args.alg_name, 
        hparams=hparams
    )


# Create clustering
def collect_clustering(args):
    create_clustering(
        args.output_file_path, 
        args.max_length, 
        args.threshold
    )


# Run unit test
def run_unit_test(args):
    unit_test(
        args.output_file_path, args.d_params_path, 
        args.originals_path, args.translate_only, 
        args.invalidate_cache, args.container_workspace
    )

# Run evaluation
def run_evaluation(args, target_KE_cluster):
    eval_results = run_eval(
        args.output_file_path, 
        args.preKE_cluster_result, 
        target_KE_cluster,
        args.max_length, 
        args.threshold, 
        args.embedding_model
    )
    return eval_results
    

def get_error_request_cluster(args):
    all_info = pd.read_csv(args.edit_sequence_path, encoding="utf-8").iloc[args.i]
    target_KE_cluster = all_info[f'layer_{args.layer_num}_{args.mlp_or_tc}']
    error_num = all_info.Error
    error_descp = all_info.brief_description
    
    print(f">>>> Target Error {error_num}: {error_descp}")
    
    return [
        {
        "prompt": all_info.prompt,
        "subject": all_info.subject,
        "target_new": {"str": all_info.target_new},
        }
    ], str(target_KE_cluster), error_num, error_descp


# Save metrics to WandB
def log_metrics(eval_results, target_KE_cluster, error_num, error_descp):
    wandb.log({
        "Error_NO.": error_num,
        "Error_description": error_descp,
        "Target_KE_Cluster": target_KE_cluster
    })
    
    if eval_results and len(eval_results) > 0:
        wandb.log({
            "SubMetrics/ori_overall_accuracy": eval_results[0],
            "SubMetrics/amount_targets": eval_results[1],
            "SubMetrics/amount_targets_new": eval_results[2],
            "SubMetrics/amount_targets_in_success": eval_results[3],
            "SubMetrics/overall_accuracy_afterKE": eval_results[4]
        })
        
        wandb.log({
            "MainMetrics/accuracy_of_target_cluster": eval_results[5],
            "MainMetrics/change_of_target_cluster": eval_results[6],
            "MainMetrics/overall_accuracy_destroy": eval_results[7],
            "MainMetrics/Specificity": eval_results[8],
            "MainMetrics/Destructiveness": eval_results[9]
        })
    
    else:
        print("Skipping metric logging.")

 
 
def main():
    args = parse_arguments()
    init_wandb(args)
    
    model, tok = load_model_and_tokenizer(args)
    hparams = set_hyperparameters(args)
    request, target_KE_group, error_num, error_descp = get_error_request_cluster(args)
 
    # Apply KE, run unit test, and evaluation
    print("args.alg_name", args.alg_name)
    apply_model_editing(model, tok, request, args, hparams)
    run_unit_test(args)

    if args.alg_name != "noKE":
        eval_results = run_evaluation(args, target_KE_group)
    else:
        eval_results = []
        collect_src_error = collect_clustering(args)

    log_metrics(eval_results, target_KE_group, error_num, error_descp)
    wandb.finish()

    print("Model editing, unit testing, and evaluation complete!")
 
if __name__ == "__main__":
    main()

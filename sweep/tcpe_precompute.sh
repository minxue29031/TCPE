#!/bin/bash

v_lr_list=(0.01 0.05 0.005)
v_num_grad_steps_list=(600)
clamp_list=(18 13 8 4)
mom2_n_samples_list=(500000) 
error_list=(6 0 4 3)   

layer_num=19
expansion_factor=4
hidden_in=4096
tc_hidden_out=$((expansion_factor * hidden_in))  



for v_lr in "${v_lr_list[@]}"; do
    for v_num_grad_steps in "${v_num_grad_steps_list[@]}"; do  
        for clamp in "${clamp_list[@]}"; do
            for mom2_n_samples in "${mom2_n_samples_list[@]}"; do
                for i in "${error_list[@]}"; do
                    echo "Running with v_lr=$v_lr, v_num_grad_steps=$v_num_grad_steps, clamp_norm_factor=$clamp, mom2_n_samples=$mom2_n_samples, i=$i"
                    CUDA_VISIBLE_DEVICES=2,3 python experiments/KE_TC_oneprompt.py \
                        --MODEL_NAME "codellama/CodeLlama-7b-Instruct-hf" \
                        --layer_num "$layer_num" \
                        --TC_layer_file "transcoder_path" \
                        --output_file_path "output/TCPE_TC${expansion_factor}/layer${layer_num}_vlr${v_lr}_vstep${v_num_grad_steps}_clamp${clamp}_mom${mom2_n_samples}_error${i}" \
                        --alg_name "TCPE_PRE" \
                        --mlp_or_tc "TC_ef${expansion_factor}" \
                        --mom2_n_samples "$mom2_n_samples" \
						--edit_sequence_path "data/request/requests_last_subject.csv" \
                        --mom2_dataset "MIN12352/Java_D_dst" \
                        --epsilon 1e-8 \
                        --v_num_grad_steps "$v_num_grad_steps" \
                        --v_lr "$v_lr" \
                        --clamp_norm_factor "$clamp" \
						--save_k_v_upd \
                        --i "$i" \
                        --container_workspace "container_workspace" \
                        --translation_batch_size 14 \
                        --preKE_cluster_result "TC_evaluation/TC_evaluation_ef${expansion_factor}/old_cluster" \
                        --save_path "TC_k_v_w/TCPE_TC${expansion_factor}/save_layer${layer_num}_vlr${v_lr}_vstep${v_num_grad_steps}_clamp${clamp}_mom${mom2_n_samples}_error${i}/"
                done
            done
        done
    done
done

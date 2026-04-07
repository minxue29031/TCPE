#!/bin/bash


layer_num=19
expansion_factor=4
hidden_in=4096
tc_hidden_out=$((expansion_factor * hidden_in))  

# Hyperparameter Sweeps
save_path_list=(
"error_1_path"
"error_2_path"
"error_3_path"
"error_4_path"
)

lim_combinations=(
"0.1 0.1 0.1 0.1"
)

upd_decay_combinations=(
"1.1 1.1 1.1 1.1"
)


for lim_list in "${lim_combinations[@]}"; do
    for upd_decay_list in "${upd_decay_combinations[@]}"; do

        decay_id=$(echo "$upd_decay_list" | sed 's/ /_/g')
		lim_id=$(echo "$lim_list" | sed 's/ /_/g')

        echo "Running with lim=$lim_list and upd_decay_list=$upd_decay_list"

        CUDA_VISIBLE_DEVICES=0 python experiments/KE_TC_allprompt.py \
            --MODEL_NAME "codellama/CodeLlama-7b-Instruct-hf" \
            --layer_num "$layer_num" \
			--TC_layer_file "transcoder_path" \
            --output_file_path "output/allprompt_layer${layer_num}_TC_ef${expansion_factor}_lim${lim_id}_decay${decay_id}" \
            --alg_name "TCPE" \
			--edit_sequence_path "data/request/requests_last_subject.csv" \
            --container_workspace "container_workspace" \
            --translation_batch_size 14 \
            --lim $lim_list \
            --save_path_list "${save_path_list[@]}" \
            --upd_decay_list $upd_decay_list \
			--mlp_or_tc "TC_ef${expansion_factor}" \
            --preKE_cluster_result "TC_evaluation/TC_evaluation_ef${expansion_factor}/old_cluster"
    done
done

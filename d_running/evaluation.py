import os
import shutil
import sys
import argparse
from base64 import b64encode
from hashlib import md5, sha1
import argparse
import torch
import wandb
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gc
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), "d_running"))
import lib.models as model_loading_saving
import cluster as cluster_creation
from align_error_cluster import TextCluster as ClusteringMapper
from lib.util import retry


def save_cluster_statistics_to_csv(clustering, current_dir):
    cluster_counts = {key: len(value) for key, value in clustering.items()}
    data = []
    
    total_clusters = len(cluster_counts)
    
    for i in range(total_clusters):
        group_key = str(i)
        if group_key in cluster_counts:
            random_description = random.choice(clustering[group_key]).get("brief_description", "None").strip()
            if not random_description:
                random_description = "None"
            data.append([group_key, cluster_counts[group_key], random_description])

    remaining_groups = [key for key in clustering.keys() if key not in map(str, range(total_clusters))]
    
    if remaining_groups:
        for group_key in remaining_groups:
            selected_entry = random.choice(clustering[group_key])
            brief_description = selected_entry.get("brief_description", "None").strip()
            if not brief_description:
                brief_description = "None"
            data.append([group_key, len(clustering[group_key]), brief_description])
    
    df = pd.DataFrame(data, columns=["Group", "functions_stats", "brief_description"])
    df["sort_order"] = df["Group"].apply(lambda x: 1 if x in ["Success", "Correctness"] else 0)
    df = df.sort_values(by=["sort_order", "functions_stats"], ascending=[True, False]).drop(columns=["sort_order"])
    
    save_stats_path = os.path.join(current_dir, "cluster_statistics.csv")
    df.to_csv(save_stats_path, index=False, encoding="utf-8")
    print(f"Cluster statistics saved to {save_stats_path}")



def create_clustering(current_dir, max_length, threshold):
    print("Create clustering:")
    clustering = cluster_creation.get_clustering(current_dir, max_length, threshold)
    cluster_creation.save_clustering(current_dir, clustering)
    cluster_creation.convert_cluster_to_csv(current_dir)
    cluster_creation.print_cluster_statistics(clustering)
    save_cluster_statistics_to_csv(clustering, current_dir)
 
def map_clustering(embedding_model, current_dir, old_cluster, max_length, threshold):
    print("Map clustering to old clusterings:")
    # remove current file (info is still saved in cluster.csv)
    clustering_before_update = cluster_creation.load_clustering(current_dir)
    current_json_path = os.path.join(current_dir, "cluster.json")
    os.remove(current_json_path)

    old_file_name = "cluster.csv"

    clustering_mapper = ClusteringMapper(embedding_model, max_length, threshold)
    clustering_mapper.process_clusters(
        ori_csv=os.path.join(old_cluster, old_file_name),
        new_csv=os.path.join(current_dir, "cluster.csv"),
        update_process_csv=os.path.join(current_dir, "update_cluster_process.csv"),
        updated_cluster_csv=os.path.join(current_dir, "updated_current_cluster.csv"),
        update_source_cluster_csv=os.path.join(
            current_dir, "update_source_cluster.csv"
        ),
    )

    # create new cluster.json with updated groups
    csv_path = os.path.join(current_dir, "updated_current_cluster.csv")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.drop(
        columns=["correct", "total"], inplace=True, errors="ignore"
    )  # errors ignore needed if columns are not present

    clustered_data = {}
    for category, group in df.groupby("rename_group"):
        if category not in ["Correctness", "Success"]:
            clustered_data[category] = group.drop(columns=["rename_group"]).to_dict(
                orient="records"
            )

    clustered_data["Correctness"] = clustering_before_update.get("Correctness", [])
    clustered_data["Success"] = clustering_before_update.get("Success", [])

    with open(current_json_path, "w", encoding="utf-8") as f:
        json.dump(clustered_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully reconstructed the cluster data to {current_json_path}")

    # cleanup
 
    clustering = cluster_creation.load_clustering(current_dir)
    cluster_creation.print_cluster_statistics(clustering)

    fig, ax = plt.subplots()
    ax.bar(
        clustering.keys(),
        [len(entry) for entry in clustering.values()],
        color="skyblue",
    )


def compute_metrics(current_dir, old_cluster, cluster_edited_last):
    print("Compute metrics:")
    old_clustering = cluster_creation.load_clustering(old_cluster)
    current_clustering = cluster_creation.load_clustering(current_dir)

    # Acc. of target cluster:
    # Perc. of correctly translated samples in targeted cluster
    target_cluster_old = old_clustering[cluster_edited_last]
    targets = [entry["path"] for entry in target_cluster_old]
    amount_targets = len(targets)

    success_cluster_new = current_clustering["Success"]
    targets_in_success = [ entry for entry in success_cluster_new if entry["path"] in targets ]
    amount_targets_in_success = len(targets_in_success)

    cluster_new = current_clustering.get(cluster_edited_last, [])
    targets_new = [entry["path"] for entry in cluster_new]
    amount_targets_new = len(targets_new)


    accuracy_of_target_cluster = amount_targets_in_success / amount_targets
    change_of_target_cluster = amount_targets_new/amount_targets

    # Specificity:
    # Perc. of samples that did not change cluster (excl. samples from targeted cluster that were corrected)
    non_target_samples_old = [
        (entry["path"], cluster_id)
        for cluster_id, entries in old_clustering.items()
        if cluster_id != cluster_edited_last
        for entry in entries
    ]

    # Count samples that did not change clusters (excluding targeted cluster)
    unchanged_samples = 0
    for entry, old_cluster_id in non_target_samples_old:
        if old_cluster_id in current_clustering:
            if entry in [ sample["path"] for sample in current_clustering[old_cluster_id] ]:
                unchanged_samples += 1

    # Count target samples that changed to non 'Success' cluster
    wrongly_changed_target_samples = 0
    current_success_cluster = [ sample["path"] for sample in current_clustering["Success"] ]
    current_target_cluster = [ sample["path"] for sample in current_clustering[old_cluster_id] ]

    for entry in old_clustering[cluster_edited_last]:
        if ( entry["path"] not in current_success_cluster and entry["path"] not in current_target_cluster ):
            wrongly_changed_target_samples += 1

    total_non_target_samples = len(non_target_samples_old)
    specificity = unchanged_samples / total_non_target_samples #( total_non_target_samples +wrongly_changed_target_samples )

    # Overall Accuracy:
    # Percentage of correctly translated samples in the new clustering
    success_cluster_new = current_clustering["Success"]
    total_samples = sum(len(cluster) for cluster in current_clustering.values())
    correctly_translated_samples = len(success_cluster_new)

    overall_accuracy = correctly_translated_samples / total_samples
 

    # Redefine Destructiveness:
    # Percentage of samples that were correct in old clustering but became faulty in new clustering
    success_cluster_old = old_clustering["Success"]
    old_correct_samples = {entry["path"] for entry in success_cluster_old}
    new_success_paths = {entry["path"] for entry in success_cluster_new}

    correct_became_faulty = old_correct_samples - new_success_paths
    destructiveness = len(correct_became_faulty) / len(old_correct_samples)
    
    succ_ori = [entry["path"] for entry in success_cluster_old]
    amount_succ_ori = len(succ_ori)
    ori_overall_accuracy = amount_succ_ori / total_samples
    overall_accuracy_destroy = overall_accuracy/ori_overall_accuracy
    
    
    return ori_overall_accuracy, amount_targets, amount_targets_new, amount_targets_in_success, overall_accuracy, accuracy_of_target_cluster, change_of_target_cluster,overall_accuracy_destroy, specificity, destructiveness
 
 

def run_eval(current_dir, old_cluster, cluster_edited_last, max_length, threshold, embedding_model): 
    create_clustering(current_dir, max_length, threshold)
    map_clustering(embedding_model, current_dir, old_cluster, max_length, threshold)
    
    (
        ori_overall_accuracy,
        amount_targets,
        amount_targets_new,
        amount_targets_in_success,
        overall_accuracy,
        accuracy_of_target_cluster,
        change_of_target_cluster,
        overall_accuracy_destroy,
        specificity,
        destructiveness,
    ) = compute_metrics(current_dir, old_cluster, cluster_edited_last)
    
    return (
        ori_overall_accuracy,
        amount_targets,
        amount_targets_new,
        amount_targets_in_success,
        overall_accuracy,
        accuracy_of_target_cluster,
        change_of_target_cluster,
        overall_accuracy_destroy,
        specificity,
        destructiveness,
    )

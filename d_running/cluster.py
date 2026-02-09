import json
import torch
import numpy as np
import os
import glob
import shutil 
import argparse
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import pandas as pd 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"

_model, _tok = None, None

def get_model_and_tokenizer():
    global _model
    global _tok
    if _model is None or _tok is None:
        _tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
    return _model, _tok

def embed_texts(texts, max_length):
    model, tok = get_model_and_tokenizer()
    inputs = tok(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
        with torch.inference_mode():
            outputs = model(**inputs.to(DEVICE)).last_hidden_state[:, 0]
    
    return F.normalize(outputs, p=2, dim=1).cpu().numpy()

def cluster(vectors, threshold):
    similarity_matrix = cosine_similarity(vectors)
    cluster_labels = [-1] * len(vectors)  
    current_cluster = 0

    for i in range(len(vectors)):
        if cluster_labels[i] != -1:
            continue
        cluster_labels[i] = current_cluster
        for j in range(i + 1, len(vectors)):
            if similarity_matrix[i][j] > threshold:
                cluster_labels[j] = current_cluster
        current_cluster += 1
    return cluster_labels

def cluster_texts(texts, max_length, threshold, **cluster_kwargs):
    embeddings = embed_texts(texts, max_length=max_length)
    return cluster(embeddings, threshold=threshold)

def convert_description(error):
    start_index = error.find("Error:")
    if start_index == -1:
        return error.strip()

    end_index = error.find("\\", start_index)
    if end_index == -1:
        end_index = len(error)
    
    return error[start_index:end_index].strip()

def cluster_errors(errors, max_length, threshold, **cluster_kwargs):
    correctness_errors = [error for error in errors if error["error"] == "Correctness"]
    other_errors = [error for error in errors if error["error"] != "Correctness"]
    assert len(correctness_errors) + len(other_errors) == len(errors)

    descriptions = [convert_description(error["description"]) for error in other_errors]
    cluster_indices = cluster_texts(descriptions, max_length=max_length, threshold=threshold, **cluster_kwargs)

    clustering = defaultdict(list)
    for cluster, error, brief_description in zip(cluster_indices, other_errors, descriptions):
        cluster = int(cluster)
        error['brief_description'] = brief_description  
        clustering[str(cluster)].append(error)

    if len(correctness_errors) > 0:
        clustering["Correctness"].extend(correctness_errors)

    return dict(clustering)

def cluster_results(results, max_length, threshold, **cluster_kwargs):
    errors = [result for result in results if not result["success"]]
    success = [result for result in results if result["success"]]
    assert len(errors) + len(success) == len(results)

    clustering = cluster_errors(errors, max_length=max_length, threshold=threshold, **cluster_kwargs)
    if len(success) > 0:
        clustering["Success"] = success

    return clustering

def save_clustering(model_path, clustering):
    with open(os.path.join(model_path, "cluster.json"), "w", encoding="utf8") as f:
        json.dump(clustering, f)

def load_clustering(model_path):
    with open(os.path.join(model_path, "cluster.json"), "r", encoding="utf8") as f:
        clustering = json.load(f)
    return clustering

def get_clustering(model_path, max_length, threshold):
    cluster_path = os.path.join(model_path, "cluster.json")
    if os.path.isfile(cluster_path):
        with open(cluster_path, "r", encoding="utf8") as f:
            return json.load(f)

    with open(os.path.join(model_path, "results.json"), "r", encoding="utf8") as f:
        results = json.load(f)

    clustering = cluster_results(results, max_length=max_length, threshold=threshold)

    return clustering

def get_cluster_counts(clustering):
    cluster_counts = {key: len(value) for key, value in clustering.items()}
    return cluster_counts 

def convert_cluster_to_csv(model_path):
    cluster_path = os.path.join(model_path, "cluster.json")
    with open(cluster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flattened_data = []
    for key, value in data.items():
        for entry in value:
            entry['group'] = key
            flattened_data.append(entry)

    df = pd.DataFrame(flattened_data)
    output_csv_path = os.path.join(model_path, "cluster.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully saved the cluster data to {output_csv_path}")

def print_cluster_statistics(clustering):
    cluster_counts = get_cluster_counts(clustering)
    total_clusters = len(cluster_counts)
    
    print(f"Total clusters: {total_clusters}\n")
    
    for i in range(total_clusters):
        group_key = str(i)
        if group_key in cluster_counts:
            print(f"Group {i}: {cluster_counts[group_key]} functions")

    remaining_groups = [key for key in clustering.keys() if key not in map(str, range(total_clusters))]
    
    if remaining_groups:
        print("\nRemaining Groups:")
        for group_key in remaining_groups:
            print(f"Group {group_key}: {len(clustering[group_key])} functions")


def main(args):
    clustering = get_clustering(args.model_path, args.max_length, args.threshold)
    save_clustering(args.model_path, clustering)
    print_cluster_statistics(clustering)    
    convert_cluster_to_csv(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--max_length", type=int, default=6, help="Maximum length for tokenization")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for clustering")

    args = parser.parse_args()
    main(args)

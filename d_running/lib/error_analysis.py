import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.cluster import HDBSCAN
from transformers import AutoTokenizer, AutoModel


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
            device_map=DEVICE,
            torch_dtype=torch.float16,
        )
    return _model, _tok


def embed_texts(texts):
    model, tok = get_model_and_tokenizer()
    inputs = tok(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
        with torch.inference_mode():
            outputs = model(**inputs.to(DEVICE)).last_hidden_state[:, 0]
    
    return F.normalize(outputs, p=2, dim=1).cpu().numpy()


def cluster(vectors, min_cluster_size=5):
    clustering = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed").fit(1 - (vectors @ vectors.T))
    return clustering.labels_


def cluster_texts(texts, **cluster_kwargs):
    embeddings = embed_texts(texts)
    return cluster(embeddings, **cluster_kwargs)


def convert_description(error):
    if error.count("'") >= 2:
        error = error[error.find("'")+1:error.rfind("'")]
        error = error.replace("\\n", "\n")
    # remove file paths / line numbers
    new = ""
    for line in error.split("\n"):
        if line.count(":") > 0:
            line = line.split(":", 1)[1]
        new += line.strip() + "\n"
        if len(new.strip()) > 0:
            break # keep only first error
    return new.strip()


def cluster_errors(errors, **cluster_kwargs):
    correctness_errors = [error for error in errors if error["error"] == "Correctness"]
    other_errors = [error for error in errors if error["error"] != "Correctness"]
    assert len(correctness_errors) + len(other_errors) == len(errors)

    descriptions = [convert_description(error["description"]) for error in other_errors]
    cluster_indices = cluster_texts(descriptions, **cluster_kwargs)

    clustering = defaultdict(list)
    for cluster, error in zip(cluster_indices, other_errors):
        cluster = int(cluster)
        if cluster == -1:
            cluster = "Noise"
        clustering[str(cluster)].append(error)
    
    if len(correctness_errors) > 0:
        clustering["Correctness"].extend(correctness_errors)

    return dict(clustering)


def cluster_results(results, **cluster_kwargs):
    errors = [result for result in results if not result["success"]]
    success = [result for result in results if result["success"]]
    assert len(errors) + len(success) == len(results)

    clustering = cluster_errors(errors, **cluster_kwargs)
    if len(success) > 0:
        clustering["Success"] = success

    return clustering


def invert_clustering(clustering):
    inverse_clustering = {}
    for key, values in clustering.items():
        for value in values:
            inverse_clustering[value["path"]] = (key, value)
    
    return inverse_clustering
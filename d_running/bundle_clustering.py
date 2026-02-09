import os
import json
import argparse
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict


def read_code(model_path, sample):
    path = os.path.join(
        "models",
        model_path,
        "translations",
        sample
    )
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def get_git_diff(sample_path, model_path_a, model_path_b):
    path_a = os.path.join(
        "models",
        model_path_a,
        "translations",
        sample_path
    )
    path_b = os.path.join(
        "models",
        model_path_b,
        "translations",
        sample_path
    )

    result = subprocess.run(
        ["git", "diff", "-U1000", "--no-index", path_a, path_b],
        capture_output=True
    )

    return result.stdout.decode("utf8")

def main(args):
    model_paths = args.model_paths
    clusterings = []
    for model_path in model_paths:
        with open(os.path.join("models", model_path, "cluster.json"), "r", encoding="utf8") as f:
            clusterings.append(json.load(f))

    print("Mapping colours...")
    colormap = plt.get_cmap("Set2")
    unique_labels = set(sum((list(clustering.keys()) for clustering in clusterings), start=[]))
    unique_labels = list(sorted(unique_labels, reverse=True, key=lambda l: len(clusterings[0][l]) if l in clusterings[0] else 0))
    colors_for_labels = {
        label: mpl.colors.rgb2hex(colormap(i % colormap.N))
        for i, label in enumerate(unique_labels)
    }
    
    print("Computing nodes...")
    nodes = []
    for i, clustering in enumerate(clusterings):
        this_nodes = []
        for key, values in sorted(clustering.items(), key=lambda t: len(t[1]), reverse=True):
            value = len(values)
            style = dict(
                color=colors_for_labels[key],
            )
            label_name = f"{key}\n{value}"
            label_position = "left"
            if 0 < i < len(clusterings) - 1:
                label_position = "inside"
                label_name = str(value)
            elif i == len(clusterings) - 1:
                label_position = "right"
            label = dict(
                position=label_position,
                formatter=label_name,
            )
            node = dict(
                name=f"{model_paths[i]}/{key}",
                value=value,
                depth=i,
                itemStyle=style,
                label=label,
            )
            
            this_nodes.append(node)
        nodes.extend(this_nodes)

    print("Computing inverse clusterings...")
    inverse_clusterings = []
    for clustering in clusterings:
        inverse_clustering = {}
        for key, values in clustering.items():
            for value in values:
                inverse_clustering[value["path"]] = key
        inverse_clusterings.append(inverse_clustering)

    print("Computing flows...")
    flows = []
    for i, clustering in enumerate(clusterings):
        this_flow = defaultdict(lambda: defaultdict(list))
        if i == 0:
            continue

        previous_inverse_clustering = inverse_clusterings[i-1]
        for key, values in tqdm(clustering.items(), desc=f"for clustering {i}"):
            for value in values:
                sample_path = value["path"]
                previous_key = previous_inverse_clustering[sample_path]
                code_before = read_code(model_paths[i-1], sample_path)
                code_after = read_code(model_paths[i], sample_path)
                diff = get_git_diff(sample_path, model_paths[i-1], model_paths[i])
                change = dict(
                    before=code_before,
                    after=code_after,
                    diff=diff,
                    name=sample_path,
                )
                this_flow[f"{model_paths[i-1]}/{previous_key}"][f"{model_paths[i]}/{key}"].append(change)
        
        this_flow_serial = []
        for key, inner_flow in this_flow.items():
            for inner_key, changes in inner_flow.items():
                this_flow_serial.append(dict(
                    source=key,
                    target=inner_key,
                    value=len(changes),
                    changes=changes,
                ))
        flows.extend(this_flow_serial)

    result = dict(
        nodes=nodes,
        links=flows,
    )

    print("Saving result...")
    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_paths", nargs="+")
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    main(args)
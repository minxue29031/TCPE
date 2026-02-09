# 🧠  TCPE: TransCoder-based Precise Editing

 
This repository implements **TCPE (TransCoder-based Precise Editing)**, a framework for **interpretable, neuron-level knowledge editing** in large language models (LLMs).

The key functionalities include:

* **TCPE Workflow**: Perform precise and interpretable knowledge edits by leveraging the sparsity and monosemanticity of TransCoder neurons.
* **KECode benchmark**: Provide a new evaluation benchmark for code-to-code translation based on functional equivalence. 


 ## ⚙️ Installation

Install dependencies:

```bash

pip install -r requirements.txt
```

## 📁 Repository Structure (Key Modules)

```
├── run_tcpe.py                   # Entry point to run TCPE experiments
├── tcpe/                         # Main TCPE workflow                   
│   ├── tcpe_main.py               
│   ├── compute_u.py           
│   └── compute_v.py          
├── d_running/                    # KECode benchmark
│   ├── cluster.py             
│   ├── evaluation.py            
│   └── unit_test.py          
├── huggingface/
│   └── transcoder_adapter.py     # Adapter for TransCoder integration
├── sae_training/
├── util/
└── requirements.txt          
```


## 🚀 Experiment Setup

1. **Generate TransCoder layer files**
   First, use the [Transcoder Circuits repository](https://github.com/jacobdunefsky/transcoder_circuits) to generate the **TransCoder for the target model layer** for interpretability analysis.

2. **Run TCPE Preprocessing**

```bash
python run_tcpe.py --MODEL_NAME <model_name> \
                   --layer_num <layer_number> \
                   --TC_layer_file <path_to_transcoder_file> \
                   --output_file_path <output_dir> \
                   --alg_name TCPE_pre \
                   --i <sequence_index> \
                   --translation_batch_size <batch_size> \
                   --preKE_cluster_result <cluster_path> \
                   --container_workspace <workspace_path> \
                   --save_k_v_upd
```

* `--alg_name TCPE_pre` → Preprocessing step before knowledge editing.

3. **Run TCPE Knowledge Editing**

```bash
python run_tcpe.py --MODEL_NAME <model_name> \
                   --layer_num <layer_number> \
                   --TC_layer_file <path_to_transcoder_file> \
                   --output_file_path <output_dir> \
                   --alg_name TCPE \
                   --lim <active_neuron_limit> \
                   --abla_exp \
                   --i <sequence_index> \
                   --translation_batch_size <batch_size> \
                   --preKE_cluster_result <cluster_path> \
                   --container_workspace <workspace_path>
```

* `--alg_name TCPE` → Apply TCPE and test neuron-level knowledge editing.
* `--lim` → Limit of active neurons for testing different activity levels.
* `--abla_exp` → Enable ablation experiment to measure intervention impact.

**Note:**

* After running TCPE, the framework automatically runs unit tests and evaluation metrics, including cluster-level accuracy, specificity, and destructiveness.
* Metrics are logged to **WandB** for analysis.


## ⚙️ Configuration Parameters

| Argument                    | Type    | Required | Description                                                    |
| --------------------------- | ------- | -------- | -------------------------------------------------------------- |
| `--MODEL_NAME`              | `str`   | Yes      | Hugging Face model identifier or local checkpoint path.        |
| `--layer_num`               | `int`   | Yes      | Target layer number for model editing.                         |
| `--TC_layer_file`           | `str`   | No       | Path to TransCoder layer file (for interpretability analysis). |
| `--output_file_path`        | `str`   | Yes      | Directory to save output and evaluation results.               |
| `--alg_name`                | `str`   | Yes      | Algorithm name: `TCPE_pre`, `TCPE`, `noKE`.                    |
| `--mlp_or_tc`               | `str`   | No       | Select covariance matrix: `MLP`, `TC_ef2`, `TC_ef3`, …         |
| `--mom2_n_samples`          | `int`   | No       | Number of samples for computing covariance matrix.             |
| `--mom2_dataset`            | `str`   | No       | Dataset for covariance computation.                            |
| `--epsilon`                 | `float` | No       | Small value to avoid zero diagonal in covariance.              |
| `--v_num_grad_steps`        | `int`   | No       | Number of gradient steps for optimization.                     |
| `--v_lr`                    | `float` | No       | Learning rate for optimization.                                |
| `--clamp_norm_factor`       | `int`   | No       | Clamp factor for weight updates.                               |
| `--edit_sequence_path`      | `str`   | No       | Path to KE request and target cluster CSV.                     |
| `--i`                       | `int`   | Yes      | Index of the specific sequence to process.                     |
| `--fact_token`              | `str`   | No       | Choose token for editing: `subject_last` or `last`.            |
| `--save_k_v_upd`            | `bool`  | No       | Save the update matrix to file.                                |
| `--save_path`               | `str`   | No       | Path to save/load key-value update matrices.                   |
| `--upd_decay`               | `float` | No       | Decay factor for weight updates.                               |
| `--lim`                     | `float` | No       | Select top-k active points for neuron-level editing.           |
| `--abla_exp`                | `bool`  | No       | Flag to perform ablation experiment.                           |
| `--container_workspace`     | `str`   | Yes      | Workspace for intermediate files.                              |
| `--d_params_path`           | `str`   | No       | Path to dataset with parameters.                               |
| `--originals_path`          | `str`   | No       | Path to original examples for evaluation.                      |
| `--translate_only  | `bool`  | No       | Only perform translation without evaluation.                   |
| `--invalidate_cache`  | `bool`  | No       | Ignore cached results.                                         |
| `--translation_batch_size`  | `int`   | Yes      | Batch size for translation / evaluation.                       |
| `--preKE_cluster_result`    | `str`   | Yes      | Path to cluster result without KE.                             |
| `--max_length`              | `int`   | No       | Number of tokens to classify errors.                           |
| `--threshold`               | `float` | No       | Similarity threshold for error classification.                 |
| `--embedding_model`         | `str`   | No       | Embedding model used for clustering evaluation.                |



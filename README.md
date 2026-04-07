#  TCPE: TransCoder-based Precise Editing

 
This repository implements **TCPE (TransCoder-based Precise Editing)**, a framework for **interpretable, neuron-level knowledge editing** in large language models (LLMs).

The key functionalities include:

* **TCPE Workflow**: Perform precise and interpretable knowledge edits by leveraging the sparsity and monosemanticity of TransCoder neurons.
* **KECode benchmark**: Provide a new evaluation benchmark for code-to-code translation based on functional equivalence. 


 ## Installation

Install dependencies:

```bash

pip install -r requirements.txt
```
 

## Experiment Setup

**<h3>1. Generate TransCoder layer files</h3>**

   First, use the [Transcoder Circuits repository](https://github.com/jacobdunefsky/transcoder_circuits) to generate the **TransCoder for the target layer** for interpretability analysis.

 
**<h3>2. Run TCPE Preprocessing</h3>**

   This step precomputes the update matrices for different types of errors, which are required for subsequent experiments.

   ```bash
   bash sweep/tcpe_precompute.sh
   ```
   **Notes:** 
   TransCoder neurons are sparse by nature — only a small subset of neurons activate for any given input. As a result, the quality of the precomputed update matrices is sensitive to the choice of hyperparameters. With poorly chosen values, the set of active neurons may be too small or even empty, yielding zero or degenerate updates. The sweep in tcpe_precompute.sh explores different hyperparameter configurations to identify settings that produce valid, non-trivial update matrices before proceeding to the editing step.

**<h3>3. Run TCPE Knowledge Editing</h3>**
 
   This step applies TCPE to perform neuron-level knowledge editing on the model.
   
   ```bash
   bash sweep/run_tcpe.sh
   ```

   **Key Arguments:**
   
   * `--alg_name TCPE` → Specifies the algorithm to use for neuron-level knowledge editing.
   * `--lim` → Sets the limit on active neurons, allowing experiments with different activity levels.
   * `--abla_exp` → Enables ablation experiments to measure the impact of interventions on the model.
   
   **Notes:** 
   * Ensure that preprocessing (Step 2) has been completed, as the generated update matrices are required.
   * Adjust `--lim` and `--abla_exp` according to your experimental setup.
 
## Acknowledgments

We gratefully acknowledge that our training code is heavily inspired by **TransCoder** ([GitHub](https://github.com/jacobdunefsky/transcoder_circuits)) and **ROME** ([GitHub](https://github.com/kmeng01/rome)). We thank the authors for their foundational work.

 

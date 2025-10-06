
# Studying the Effect of Calibration Data Selection on Pruned Large Language Models

## Folder Structure

The main directory contains three folders:

- `wanda`: Contains the source code of the wanda pruning method, along with the modifications made for this project.
- `analyze data`: Contains the scripts used to analyze the results.

---


## Creating the Environment
To create the conda environment, run the following command:

```bash
conda env create -f environment.yml
```

---

## Executing the Code

To execute the code, first activate the conda environment:

```bash
conda activate wanda
```

Then, navigate to the `wanda` directory:

```bash
cd wanda
```


Execute the main script with the desired parameters. For example:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --model Qwen/Qwen2.5-7B --prune_method wanda --sparsity_ratio 0.9 --sparsity_type unstructured --nsamples 128 --calibration_dataset c4 --evaluate_sparse
```

---

## List of the most important parameters

- `--model`: Specifies the model to be pruned and evaluated. Example: `Qwen/Qwen2.5-7B`.
- `--prune_method`: Specifies the pruning method to be used. Example: `wanda`.
- `--sparsity_ratio`: Specifies the sparsity ratio for pruning. Example: `0.9`.
- `--sparsity_type`: Specifies the type of sparsity. Example: `unstructured`.
- `--nsamples`: Specifies the number of samples to use for calibration. Example: `128`.
- `--calibration_dataset`: Specifies the dataset to use for calibration. Example: `c4`.
- `--evaluate_sparse`: Flag to indicate whether to evaluate the sparse model.
- `--evaluate_dense`: Flag to indicate whether to evaluate the dense model. 
- `--calibration_mode`: Specifies the method for joining different calibration datasets. Options include `concat`, `random_sample`, `prototype`, `most_different`, `prototype_iou`, `most_different_iou`, `prototype_st`, and `most_different_st`. Default is `concat`.


---

## Notebook used for the analysis of the results

For executing the notebooks, the creation of another conda environment is recommended. You can create it using the following command:

```bash
conda env create -f environment_notebooks.yml
```


Not all nobooks are functional since they need all the data points gathered during the experiments. The data was not fully uploaded to the repository due to its size.
Still, the notebooks are already executed and the results are shown within each file.


---
## Main References

The work presented here is based on the following paper:
- [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- [Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning](https://arxiv.org/abs/2410.07461)


This code is build apon the repository of the works cited above. Please refer to their repository for more details on the pruning method and the original implementation.
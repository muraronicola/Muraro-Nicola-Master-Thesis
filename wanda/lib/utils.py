import os
import numpy as np
import random
import torch
import time


def save_config_file(config_filename, args):
    
    with open(config_filename, 'w') as f:
        f.write(f"model: {args.model}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"nsamples: {args.nsamples}\n")
        f.write(f"sparsity_ratio: {args.sparsity_ratio}\n")
        f.write(f"sparsity_type: {args.sparsity_type}\n")
        f.write(f"prune_method: {args.prune_method}\n")
        f.write(f"cache_dir: {args.cache_dir}\n")
        f.write(f"use_variant: {args.use_variant}\n")
        f.write(f"save_model: {args.save_model}\n")
        f.write(f"eval_zero_shot: {args.eval_zero_shot}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"input_format: {args.input_format}\n")
        f.write(f"calibration_dataset: {args.calibration_dataset}\n")
        f.write(f"ensamble_dataset: {args.ensamble_dataset}\n")
        f.write(f"only_one_kind: {args.only_one_kind}\n")
        f.write(f"padding_side: {args.padding_side}\n")
        f.write(f"data_seqlen: {args.data_seqlen}\n")
        f.write(f"rationale: {args.rationale}\n")
        f.write(f"num_incontext: {args.num_incontext}\n")
        f.write(f"num_cot_steps: {args.num_cot_steps}\n")
        f.write(f"verbose: {args.verbose}\n")
        f.write(f"eval_rationale: {args.eval_rationale}\n")
        f.write(f"local_tests: {args.local_tests}\n")
        f.write(f"evaluate_dense: {args.evaluate_dense}\n")
        f.write(f"evaluate_sparse: {args.evaluate_sparse}\n")
        f.write(f"save_mask: {args.save_mask}\n")
        f.write(f"batch_size: {args.batch_size}\n")        
        f.close()
        

def avoid_concurrence_issues():
    #sleep for a random number of seconds to avoid concurrency issues
    time.sleep(random.uniform(1, 40))


def find_id_run(prefix, suffix):
    i = 0
    filename = f"{prefix}{i}{suffix}"
    
    while os.path.exists(filename):
        i += 1
        filename = f"{prefix}{i}{suffix}"

    return i


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_folder_structure():
    create_folder("./out/")
    create_folder("./out/cd/")
    create_folder("./out/config/")
    create_folder("./out/weight_activation/")
    create_folder("./out/out_model/")


def get_out_file_path(id_run):
    result_filename = "./out/result_{}.csv".format(id_run)
    config_filename = "./out/config/conf_{}.txt".format(id_run) 
    model_filename = "./out/out_model/model_{}.pt".format(id_run)
    
    return result_filename, config_filename, model_filename


def fix_seed(seed):
    # Setting seeds for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



def get_model_dtype(model_name):
    if model_name== "mistralai/Mistral-7B-v0.1" or model_name == "Qwen/Qwen-7B" or model_name == "meta-llama/Llama-2-7b-hf" or model_name == "Qwen/Qwen2.5-7B":
        return torch.bfloat16
    else: 
        return torch.float16



def get_batch_size_dictionary(model, batch_size):
    dizionario = {
        "microsoft/phi-2": {
            "gsm8k": 128,
            "svamp": 128,
            "mawps": 128,
            "esnli": 64,
            "anli_r1": 32,
            "rte": 64,
            "boolq": 32,
            "commonsense_qa": 128,
            "race": 16,
            "winogrande": 256,
            "wmt14": 128,
            "iwslt": 128,
            "opc": 64,
            "ds1000": 32,
            "mbpp": 512
        },
        "mistralai/Mistral-7B-v0.1": {
            "gsm8k": 128,
            "svamp": 512,
            "mawps": 256,
            "esnli": 128,
            "anli_r1": 32,
            "rte": 128,
            "boolq": 32,
            "commonsense_qa": 256,
            "race": 16,
            "winogrande": 256,
            "wmt14": 128,
            "iwslt": 128,
            "opc": 128,
            "ds1000": 32,
            "mbpp": 512
        },
        "meta-llama/Llama-2-7b-hf": {
            "gsm8k": 64,
            "svamp": 64,
            "mawps": 64,
            "esnli": 64,
            "anli_r1": 16,
            "rte": 64,
            "boolq": 16,
            "commonsense_qa": 64,
            "race": 8,
            "winogrande": 128,
            "wmt14": 64,
            "iwslt": 64,
            "opc": 64,
            "ds1000": 16,
            "mbpp": 64
        },
        "baichuan-inc/Baichuan-7B": {
            "gsm8k": 32,
            "svamp": 32,
            "mawps": 32,
            "esnli": 32,
            "anli_r1": 8,
            "rte": 32,
            "boolq": 8,
            "commonsense_qa": 64,
            "race": 4,
            "winogrande": 64,
            "wmt14": 32,
            "iwslt": 32,
            "opc": 16,
            "ds1000": 8,
            "mbpp": 32
        },
        "Qwen/Qwen-7B": {
            "gsm8k": 64,
            "svamp": 64,
            "mawps": 64,
            "esnli": 64,
            "anli_r1": 16,
            "rte": 32,
            "boolq": 8,
            "commonsense_qa": 64,
            "race": 8,
            "winogrande": 128,
            "wmt14": 64,
            "iwslt": 64,
            "opc": 32,
            "ds1000": 16,
            "mbpp": 128
        }
    }

    
    default = {
        "gsm8k": min(32, batch_size),
        "svamp": min(32, batch_size),
        "mawps": min(32, batch_size),
        "esnli": min(32, batch_size),
        "anli_r1": min(8, batch_size),
        "rte": min(32, batch_size),
        "boolq": min(8, batch_size),
        "commonsense_qa": min(32, batch_size),
        "race": min(2, batch_size),
        "winogrande": min(32, batch_size),
        "wmt14": min(32, batch_size),
        "iwslt": min(32, batch_size),
        "opc": min(8, batch_size),
        "ds1000": min(4, batch_size),
        "mbpp": min(8, batch_size)
    }

    if model in dizionario:
        return dizionario[model]
    else:
        return default
    
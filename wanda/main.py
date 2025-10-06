import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from importlib.metadata import version
import time
import pandas as pd
import random

from lib.prune import prune_wanda, check_sparsity
from lib.eval import eval_ppl, eval_acc, eval_bleu, eval_code

from lib.utils import save_config_file, find_id_run, create_folder_structure, get_out_file_path, fix_seed, get_model_dtype, get_batch_size_dictionary, avoid_concurrence_issues

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())




def get_llm(model_name, cache_dir="llm_weights"):
    
    model_dtype = get_model_dtype(model_name)
    print("loading model in ", model_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples taken from every dataset.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', action='store_true')

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument('--input_format', type=str, default='concat', choices=['autoregressive', 'single', 'concat', 'zero'])
    parser.add_argument("--calibration_dataset", type=str, choices=['c4', 'wikitext2', 'oscar', 'redpajama', 'pile', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2',
                'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race',
                'winogrande', 'wmt14', 'iwslt',
                'opc', 'ds1000', 'mbpp',
                'ellipses', 'random', 
                'all', 'all_no_general', 'all_no_aritm', 'all_no_nlu', 'all_no_commonsense', 'all_no_translation', 'all_no_coding'],
                nargs='+')
    parser.add_argument("--ensamble_dataset", action='store_true')
    parser.add_argument("--only_one_kind", type=str, default="no", choices=["no", "general", "aritm", "nlui", "commonsense", "translation", "nonsensical"])
    
    
    #Boh..
    parser.add_argument('--padding_side', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--data_seqlen', type=int, default=None)
    parser.add_argument('--rationale', action='store_true')
    parser.add_argument('--num_incontext', type=int)
    parser.add_argument('--num_cot_steps', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--eval_rationale', action='store_true')
    
    
    parser.add_argument('--local_tests', action='store_true')
    parser.add_argument('--evaluate_dense', action='store_true')
    parser.add_argument('--evaluate_sparse', action='store_true')
    parser.add_argument('--save_mask', action='store_true')
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--calibration_mode', type=str, default='concat', choices=['concat', 'random_sample', 'prototype', 'most_different', 'prototype_iou', 'most_different_iou', 'prototype_st', 'most_different_st'])
    parser.add_argument('--calibration_distance', type=str, default='flatten', choices=['flatten', 'mean'])
    parser.add_argument('--save_calibration_distribution', action='store_true')
    parser.add_argument('--n_calibration_sample', type=int, default=128, help='Total number of calibration samples used')
    parser.add_argument('--count_number_occurrence',  action='store_true')
    
    parser.add_argument('--exit_after_calibrationloading', action='store_true', help='Load the calibration distribution from file')
    
    
    
    ppl_datasets = ['c4', 'wikitext2', 'redpajama', 'oscar', 'pile'] 
    acc_datasets = ['gsm8k', 'svamp', 'mawps', 'esnli', 'anli_r1', 'rte', 'boolq', 'commonsense_qa', 'race', 'winogrande'] 
    bleu_datasets = ['wmt14', 'iwslt']
    code_datasets = ['opc', 'ds1000', 'mbpp']
    
    
    general_datasets = ['c4', 'wikitext2', 'oscar', 'redpajama', 'pile']
    aritm_reasoning = ['gsm8k', 'svamp', 'mawps'] #['svamp', 'mawps'] #['gsm8k', 'svamp', 'mawps']
    nlu_inference = ['esnli', 'anli_r1', 'rte', 'boolq']
    commonsense_qa = ['commonsense_qa', 'race', 'winogrande']
    translation = ['wmt14', 'iwslt']
    coding = ['opc', 'ds1000', 'mbpp']
    
    nonsensical = ['ellipses', 'random']
    
    
    
    args = parser.parse_args()
    
    assert args.n_calibration_sample <= args.nsamples, "n_calibration_sample must be less or equal than nsamples"
    #if "al" in args.calibration_dataset:
    #    assert len(args.calibration_dataset) == 1, "calibration dataset 'all option' must be the only one in the list" # Testo 'al' e non 'all' altrimemento non funziona

    device = args.device
    calibration_dataset = args.calibration_dataset
    batch_size = args.batch_size
    
    avoid_concurrence_issues()
    create_folder_structure()
    id_run = find_id_run("./out/config/conf_", ".txt")
    result_filename, config_filename, model_filename = get_out_file_path(id_run)
    save_config_file(config_filename, args)
    results = pd.DataFrame()
    
    print(f"run id {id_run}", flush=True)
    
    print("batch_size", batch_size)
    
    batch_size = get_batch_size_dictionary(args.model, batch_size)
    
    print("batch_size dictionary", batch_size)
    
    all_mode = False
    
    if len(calibration_dataset) == 1:
        if "all" == calibration_dataset[0]:
            calibration_dataset = general_datasets + aritm_reasoning + nlu_inference + commonsense_qa + translation + coding
            all_mode = True
        elif "all_no_general" == calibration_dataset[0]:
            calibration_dataset = aritm_reasoning + nlu_inference + commonsense_qa + translation + coding
            all_mode = True
        elif "all_no_aritm" == calibration_dataset[0]:
            calibration_dataset = general_datasets + nlu_inference + commonsense_qa + translation + coding
            all_mode = True
        elif "all_no_nlu" == calibration_dataset[0]:
            calibration_dataset = general_datasets + aritm_reasoning + commonsense_qa + translation + coding
            all_mode = True
        elif "all_no_commonsense" == calibration_dataset[0]:
            calibration_dataset = general_datasets + aritm_reasoning + nlu_inference + translation + coding
            all_mode = True
        elif "all_no_translation" == calibration_dataset[0]:
            calibration_dataset = general_datasets + aritm_reasoning + nlu_inference + commonsense_qa + coding
            all_mode = True
        elif "all_no_coding" == calibration_dataset[0]:
            calibration_dataset = general_datasets + aritm_reasoning + nlu_inference + commonsense_qa + translation
            all_mode = True
    
    print("calibration dataset:", calibration_dataset)
    seqlen = 2048 

    fix_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)

    print(f"model sequence length {model.seqlen} loaded")
    print("args.use_variant", args.use_variant)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side=args.padding_side, trust_remote_code=True) #, cache_dir=args.cache_dir)
    
    if tokenizer.pad_token_id is None: # and "Qwen" not in args.model:
        
        #if args.model:
        #    tokenizer.special_tokens['<extra_0>']
        if "Qwen" in args.model:
            tokenizer.pad_token = "<|endoftext|>"
        else:
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
        
    
    #if "Llama-3" in args.model: #True?
    if True: #model.generation_config.pad_token_id is None: # and "Qwen" not in args.model:
        
        print("Generation part: adding pad token to model (aka llama-3 mode on)")
        
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(model.config.vocab_size + 1)
    
    print("use device ", device)
    
    ################################################################

    if args.evaluate_dense:
        print("*"*30)
        print("evaluation with sparsity = 0")

        evaluation_seqlen = 2048
        
        model = model.to(device)
        model.eval()
        
        for dataset in ppl_datasets:
            ppl = eval_ppl(model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"Perplexity on {dataset}: {ppl}", flush=True)
            new_result = pd.DataFrame([[args.model, 0, None, dataset, None, None, ppl, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1'])
            results = pd.concat([results, new_result], ignore_index=True)    
            results.to_csv(result_filename, index=False)

        for dataset in acc_datasets:
            acc = eval_acc(args, model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"Acc. on {dataset}: {acc}", flush=True)
            new_result = pd.DataFrame([[args.model, 0, None, dataset, acc, None, None, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1']) 
            results = pd.concat([results, new_result], ignore_index=True)    
            results.to_csv(result_filename, index=False)

        for dataset in bleu_datasets:
            bleu = eval_bleu(args, model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)['bleu']
            print(f"bleu on {dataset}: {bleu}", flush=True)
            new_result = pd.DataFrame([[args.model, 0, None, dataset, None, bleu, None, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1'])
            results = pd.concat([results, new_result], ignore_index=True)
            results.to_csv(result_filename, index=False)
            
        for dataset in code_datasets:
            f1_score = eval_code(args, model, tokenizer, evaluation_seqlen, device=device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"F1. on {dataset}: {f1_score}", flush=True)
            new_result = pd.DataFrame([[args.model, 0, None, dataset, None, None, None, f1_score]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1']) 
            results = pd.concat([results, new_result], ignore_index=True)
            results.to_csv(result_filename, index=False)

        results.to_csv(result_filename, index=False)
        print("*"*30)
        
    
    ################################################################


    if args.sparsity_ratio != 0: #True for debbugging various stuff
        print("pruning starts")
        model = model.to("cpu")
        
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, seqlen, device, prune_n=prune_n, prune_m=prune_m, calibration_dataset=calibration_dataset, id_run=id_run)


    ################################################################
    
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    
    ################################################################
    
    if args.save_model:
        model.save_pretrained(model_filename)
        tokenizer.save_pretrained(model_filename)

    ################################################################
    
    if args.evaluate_sparse:
        print("*"*30)
        print("final evaluation with sparsity = {}".format(sparsity_ratio))
        
        evaluation_seqlen = 2048
        
        model = model.to(device)
        model.eval()
        
        calibration_dataset_name = calibration_dataset
        if type(calibration_dataset) == list and len(calibration_dataset) == 1:
            calibration_dataset_name = calibration_dataset[0]
            
        if all_mode:
            calibration_dataset_name = args.calibration_dataset[0] + "_" + args.calibration_mode + "_" + args.calibration_distance
        
        print("calibration dataset name:", calibration_dataset_name)

        for dataset in ppl_datasets:
            ppl = eval_ppl(model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"Perplexity on {dataset}: {ppl}", flush=True)
            new_result = pd.DataFrame([[args.model, sparsity_ratio, calibration_dataset_name, dataset, None, None, ppl, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1'])
            results = pd.concat([results, new_result], ignore_index=True)   
            results.to_csv(result_filename, index=False)
        
        for dataset in acc_datasets:
            acc = eval_acc(args, model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"Acc. on {dataset}: {acc}", flush=True)
            new_result = pd.DataFrame([[args.model, sparsity_ratio, calibration_dataset_name, dataset, acc, None, None, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1']) 
            results = pd.concat([results, new_result], ignore_index=True)    
            results.to_csv(result_filename, index=False)

        for dataset in bleu_datasets:
            bleu = eval_bleu(args, model, tokenizer, evaluation_seqlen, device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)['bleu']
            print(f"bleu on {dataset}: {bleu}", flush=True)
            new_result = pd.DataFrame([[args.model, sparsity_ratio, calibration_dataset_name, dataset, None, bleu, None, None]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1'])
            results = pd.concat([results, new_result], ignore_index=True)
            results.to_csv(result_filename, index=False)
            
        for dataset in code_datasets:
            f1_score = eval_code(args, model, tokenizer, evaluation_seqlen, device=device, dataset=dataset, batch_size=batch_size, verbose=args.verbose)
            print(f"F1. on {dataset}: {f1_score}", flush=True)
            new_result = pd.DataFrame([[args.model, sparsity_ratio, calibration_dataset_name, dataset, None, None, None, f1_score]], columns=['model', 'sparsity', 'pruning_data', 'dataset', 'acc', 'bleu', 'ppl', 'f1']) 
            results = pd.concat([results, new_result], ignore_index=True)
            results.to_csv(result_filename, index=False)

        results.to_csv(result_filename, index=False)
        print("*"*30)
    
    ################################################################

if __name__ == '__main__':
    main()

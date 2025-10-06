import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import pandas as pd
import os
import numpy as np
from .preprocess_data import get_processed_raw_weights_distribution, get_processed_raw_weights_value
from .prepare_calibration import prepare_calibration
from .ablate import AblateGPT 
import gc

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, seqlen, pad_token, local_tests=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    layers = model.model.layers
    
    n_sample = len(dataloader)
    #QWenLMHeadModel
    
    device = "cpu"
    

    #device = "cpu" if local_tests else "cuda:0"
    #model = model.to(device)
    
    # dev = model.hf_device_map["model.embed_tokens"]
    #if "model.embed_tokens" in model.hf_device_map:
    #    device = model.hf_device_map["model.embed_tokens"]
    
    #if device != "cpu":
    #    device = model.hf_device_map["model.embed_tokens"]
    
    dtype = next(iter(model.parameters())).dtype
    inps = [torch.zeros((seqlen, model.config.hidden_size), dtype=dtype, device=device, requires_grad=False) for i in range(n_sample)]
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    
    print("input allocated")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
                
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            att_mask = torch.tensor([[0 if batch[0][0][j] == pad_token else 1 for j in range(batch[0].shape[1])]]).to(device).type(torch.LongTensor)
            model(batch[0], attention_mask=att_mask) #.to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = [torch.zeros((seqlen, model.config.hidden_size), dtype=dtype, device=device, requires_grad=False) for i in range(n_sample)]
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    
    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(args, model, tokenizer, seqlen, device=torch.device("cuda:0"), prune_n=0, prune_m=0, calibration_dataset="c4", mv=False, id_run=-1):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    model_str_abbrevation = str(args.model).split("/")[1]

    print("loading calibdation data")
    
    dataloader = []
    for calib_dataset in calibration_dataset:
        print("loading calibration dataset {} ...".format(calib_dataset), end=" ")
        new_data, _, pad_token = get_loaders(calib_dataset, train=True, nsamples=args.nsamples, seed=args.seed,
                                                seqlen=seqlen, data_seqlen=args.data_seqlen, tokenizer=tokenizer, 
                                                rationale=args.rationale, input_format=args.input_format,
                                                padding_side=args.padding_side, verbose=args.verbose, num_incontext=args.num_incontext, num_cot_steps=args.num_cot_steps)
        
        dataloader.append(new_data)
        print("   done")
    
    if not mv:
        dataloader = prepare_calibration(model, dataloader, nsamples=args.n_calibration_sample, type=args.calibration_mode, distance=args.calibration_distance, save_calibration_distribution=args.save_calibration_distribution, model_name=model_str_abbrevation, dataset_name=calibration_dataset, count_number_occurrence=args.count_number_occurrence, tokenizer=tokenizer)
        if args.exit_after_calibrationloading:
            print("EXIT 0: Exiting after calibration loading as requested\n\n")
            exit(0)
    else:
        raise NotImplementedError("Majority-voting pruning not implemented yet")
    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, seqlen, pad_token, args.local_tests)
        
        #print("inps.shape", inps.shape, flush=True)
        #print("outs.shape", outs.shape, flush=True)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        if position_ids is not None:
            position_ids = position_ids.to(device)
        
        if not args.local_tests:
            inps = [inp.to(device) for inp in inps]
            outs = [ou.to(device) for ou in outs]
            model = model.to(device)
    
    print("\nready to prune\n")
    
    layers = model.model.layers
    
    filename_csv = "./out/weight_activation/specs_run__{}_{}_{}.csv".format(model_str_abbrevation, args.sparsity_ratio, args.calibration_dataset) #find_file_name("./out/weight_activation/specs_run", ".csv")
    results = pd.DataFrame() #columns=["layer", "name", "name_id", "shape[0]", "shape[1]", "sparsity", "model"])
    weight_distriution = pd.DataFrame() #columns=['model', 'layer', 'components', 'sparsity', 'value', 'counter', 'total'])
    weight_value = pd.DataFrame() #columns=['model', 'layer', 'components', 'sparsity', 'value', 'value_std'])
    
    
    for i in range(len(layers)):
        print("\n---------------------")
        print("Layer: ", i)
        
        layer = layers[i]
        subset = find_layers(layer)

        #if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        #if device != "cpu":
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        layer = layer.to(device)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        
        for j in range(len(inps)):
            with torch.no_grad():
                if args.local_tests:
                    outs[j] = outs[j].to(device)
                    inps[j] = inps[j].to(device)
                
                #if attention_mask is not None:
                if position_ids is not None:
                    outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
                else:
                    outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
                #else:
                #    outs[j] = layer(inps[j], position_ids=position_ids)[0]
                
                if args.local_tests:
                    outs[j] = outs[j].to("cpu")
                    inps[j] = inps[j].to("cpu")
        
        for h in handles:
            #print(f"removing hook {h}")
            h.remove()
            

        name_id = 0
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            
            name_id += 1
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True) #Set true at the indices of the top-k elements 
            
            
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            
            if args.save_mask:
                filename_Wmask = "./out/weight_activation/W_mask_{}_{}_{}_{}_{}.npy".format(model_str_abbrevation, args.sparsity_ratio, args.calibration_dataset[0], i, name_id)
                numpy_W_mask = compress_byte_representation(W_mask)
                np.save(filename_Wmask, numpy_W_mask)
            
            if args.save_weights:
                new_weight_distribution = get_processed_raw_weights_distribution(model_str_abbrevation, i, name_id, args.sparsity_ratio, W_metric)
                weight_distriution = pd.concat([weight_distriution, new_weight_distribution])
                new_weight_value = get_processed_raw_weights_value(model_str_abbrevation, i, name_id, args.sparsity_ratio, W_metric)
                weight_value = pd.concat([weight_value, new_weight_value])
            
            results = pd.concat([results, pd.DataFrame({"layer": [i], "name": [name], "name_id": [name_id], "shape[0]": [W_mask.shape[0]], "shape[1]": [W_mask.shape[1]], "sparsity":[args.sparsity_ratio], "model": [args.model]})], ignore_index=True)

        del indices
        del sort_res
        del W_metric
        del W_mask
        for j in range(len(inps)):
            with torch.no_grad():
                if args.local_tests:
                    outs[j] = outs[j].to(device)
                    inps[j] = inps[j].to(device)
                
                if position_ids is not None:
                    outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
                else:
                    outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
        
                if args.local_tests:
                    outs[j] = outs[j].to("cpu")
                    inps[j] = inps[j].to("cpu")
        
        inps, outs = outs, inps
        
        layer.to("cpu")


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    if args.save_mask:
        results.to_csv(filename_csv, index=False)
        
    if args.save_weights:
        weight_distriution.to_csv("./out/weight_activation/weight_distribution_{}_{}_{}.csv".format(model_str_abbrevation, args.sparsity_ratio, args.calibration_dataset[0]), index=False)
        weight_value.to_csv("./out/weight_activation/weight_value_{}_{}_{}.csv".format(model_str_abbrevation, args.sparsity_ratio, args.calibration_dataset[0]), index=False)

def compress_byte_representation(W_mask):
    numpy_W_mask = np.array(W_mask.detach().cpu(), dtype=bool)
    compressed_numpy_representation = np.packbits(numpy_W_mask, axis=None)
    
    return compressed_numpy_representation

import pandas as pd
import numpy as np
import torch


def get_processed_raw_weights_distribution(model, layer, component, sparsity,  W_metric):
    weight_distriution = pd.DataFrame() #columns=['model', 'layer', 'components', 'sparsity', 'value', 'counter', 'total'])
    
    num_bins = 1000
    value_range = (0, 1) 

    weights = W_metric.to('cpu')
    weights = torch.abs(torch.flatten(weights)).to(torch.float16)
    
    weights_np = weights.numpy()
    
    histogram, _ = np.histogram(weights_np, bins=num_bins, range=value_range)
    for i in range(num_bins):
        value = value_range[0] + (value_range[1] - value_range[0]) * (i + 0.5) / num_bins
        
        new_result = pd.DataFrame({
            'model': [model],
            'layer': [layer],
            'components': [component],
            'sparsity': [sparsity],
            'value': [value],  # Calculate bin center
            'counter': [histogram[i]],
            'total': [len(weights_np)],
            'mode': 'high_precision'
        })
        weight_distriution = pd.concat([weight_distriution, new_result])
        
    
    
    num_bins = 100
    value_range = (0, 100) 
    
    histogram, _ = np.histogram(weights_np, bins=num_bins, range=value_range)
    for i in range(num_bins):
        value = value_range[0] + (value_range[1] - value_range[0]) * (i + 0.5) / num_bins
        
        new_result = pd.DataFrame({
            'model': [model],
            'layer': [layer],
            'components': [component],
            'sparsity': [sparsity],
            'value': [value],  # Calculate bin center
            'counter': [histogram[i]],
            'total': [len(weights_np)],
            'mode': 'high_range'
        })
        weight_distriution = pd.concat([weight_distriution, new_result])
    
    return weight_distriution



def get_processed_raw_weights_value(model, layer, component, sparsity,  W_metric):
    weight_value = pd.DataFrame() #columns=['model', 'layer', 'components', 'sparsity', 'value', 'value_std'])
    
    weights = W_metric.to('cpu')
    weights = torch.abs(torch.flatten(weights)).to(torch.float16)
    
    weights_np = np.array(weights, dtype=np.float64)
    
    if sparsity != 0:
        sparsity_mask = weights_np >= np.quantile(weights_np, sparsity)
    else:
        sparsity_mask = np.ones_like(weights_np, dtype=bool)
    
    weights_sparse = weights_np[sparsity_mask]
    
    value = np.mean(weights_sparse)
    value_std = np.std(weights_sparse)
    
    new_result = pd.DataFrame({
        'model': [model],
        'layer': [layer],
        'components': [component],
        'sparsiry': [sparsity],
        'value': [value],
        'value_std': [value_std]
    })
    weight_value = pd.concat([weight_value, new_result])
    
    return weight_value


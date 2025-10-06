import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

def embedd_data(dataset, model, device="cuda:0"):
    model.eval()
    
    embedding_layer = model.model.embed_tokens
    embedding_layer.to(device)
    
    with torch.no_grad():
        #print("Embedding data...", flush=True)
        
        data_batch = []
        for i in range(len(dataset)):
            #print("sample", i)
            data_batch.append(dataset[i][0])
        
        #data_batch = [dataset[i][0].to(device) for i in range(len(dataset))]
        
        #Create a single tensor:
        data_batch = torch.cat(data_batch, dim=0)
        #print("To device...", flush=True)
        data_batch = data_batch.to(device)
        
        #print("Data batch shape = {}".format(data_batch.shape), flush=True)

        embedding = embedding_layer(data_batch)
        embedding = embedding.to("cpu")

        #print("Embedding done, shape = {}".format(embedding.shape), flush=True)

    embedding_layer.to("cpu")
    
    return embedding


def cosine_similarity_vectorized(data):
    # Normalize the data along the vector dimension
    data_normalized = F.normalize(data.to(torch.float32), p=2, dim=1)
    
    # Compute cosine similarity matrix using matrix multiplication
    similarity_matrix = torch.matmul(data_normalized, data_normalized.T)
    return similarity_matrix


def get_cosine_similarity(last_hidden_state_array_torch, distance='flatten'):
    if distance == 'flatten':
        Matrix_embeddings = last_hidden_state_array_torch.view(last_hidden_state_array_torch.shape[0], -1) # flatten the sequence length -> (batch_size, embedding_dim * sequence_length)
    elif distance == 'mean':
        Matrix_embeddings = torch.mean(last_hidden_state_array_torch, dim=1) #mean over the sequence length (batch_size, embedding_dim)
    
    cosine_similarit_matrix = cosine_similarity_vectorized(Matrix_embeddings)
    
    return cosine_similarit_matrix




def random_sample(dataloader, sample_per_dataset):
    calibration_data = []
    for dataset in dataloader:
        if len(dataset) > sample_per_dataset:
            indices = torch.randperm(len(dataset))[:sample_per_dataset]
            sampled_data = [dataset[i] for i in indices]
        else:
            sampled_data = dataset
        
        calibration_data.append(sampled_data)
    
    return torch.utils.data.ConcatDataset(calibration_data)


def use_embedding_for_sampling(dataloader, model, sample_per_dataset, distance, type='prototype', save_calibration_distribution=False, model_name="model", dataset_name="dataset"):
    calibration_data = []
    
    filename = "./out/cd/cd_{}_{}_{}.pt"
    
    for indice, dataset in enumerate(dataloader):
        print("\nDataset {}".format(indice), flush=True)
        last_hidden_state_array_torch = embedd_data(dataset, model, device="cuda:0")
        #print("embedding done", flush=True)
        cosine_similarity_matrix = get_cosine_similarity(last_hidden_state_array_torch, distance=distance)
        #print("cosine similarity done", flush=True)
        
        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(cosine_similarity_matrix, dim=1)
        #print("mean done", flush=True)
        
        if save_calibration_distribution:
            torch.save(cosine_similarity_matrix.to(torch.float16), filename.format(model_name, dataset_name[indice], distance))
        
        if type == 'prototype':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == 'most_different':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)
        
        #print("ordering done", flush=True)
        
        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])
            
    return calibration_data



def get_intersection_over_union(data_torch_tokenized, count_number_occurrence=False):
    result_matrix = torch.zeros((len(data_torch_tokenized), len(data_torch_tokenized)))
    
    unique_list = []
    count_unique_list = []
    
    for i in range(len(data_torch_tokenized)):
        unique, counts = torch.unique(data_torch_tokenized[i], return_counts=True)
        unique_list.append(unique)
        count_unique_list.append(counts)

    for i in range(len(data_torch_tokenized)):
        for j in range(i + 1, len(data_torch_tokenized)):

            #print(f"Comparing sample {i} with sample {j}")

            unique_a, count_unique_a = unique_list[i], count_unique_list[i]
            unique_b, count_unique_b = unique_list[j], count_unique_list[j]

            common_elements = torch.isin(unique_a, unique_b, assume_unique=True)

            if count_number_occurrence:
                intersection_number = 0
                for index, is_common in enumerate(common_elements):
                    if is_common:
                        element = unique_a[index]
                        index_element_a = (unique_a == element).nonzero(as_tuple=False)
                        index_element_b = (unique_b == element).nonzero(as_tuple=False)

                        this_int = min(count_unique_a[index_element_a].item(), count_unique_b[index_element_b].item())
                        intersection_number += this_int
                
                union_number = 2048 + 2048
            else:
                intersection_number = torch.sum(common_elements).item()
                union_number = unique_a.numel() + unique_b.numel() - intersection_number

            iou = intersection_number / union_number
            result_matrix[i, j] = iou
            result_matrix[j, i] = iou
            
    return result_matrix



def use_embedding_for_sampling_iou(dataloader, model, sample_per_dataset, count_number_occurrence, type='prototype_iou', save_calibration_distribution=False, model_name="model", dataset_name="dataset"):
    calibration_data = []
    
    for indice, dataset in enumerate(dataloader):
        print("\nDataset {}".format(indice), flush=True)
        data_0 = [d[0] for d in dataset]
        iou_similarity_matrix = get_intersection_over_union(data_0, count_number_occurrence=count_number_occurrence)
        
        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(iou_similarity_matrix, dim=1)
        
        #print(f"Dataset {indice} - Mean IoU: {mean_cosine_similarity.mean().item()}")  # Debugging output
        #print(torch.sort(mean_cosine_similarity, descending=True)[:100])  # Debugging output
        
        if type == 'prototype_iou':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == 'most_different_iou':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)
        
        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])
            
    return calibration_data




def get_similarity_matrix_with_st(data, tokenizer, device="cuda:0"):
    model_st = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    model_st.eval()
    model_st.to(device)

    values = []
    for index, item in enumerate(data):
        #print("decode", item)
        de_tokenized_phrase = tokenizer.decode(item[0]) #, skip_special_tokens=True)
        embedding = model_st.encode(de_tokenized_phrase, convert_to_tensor=True)
        
        values.append(embedding)

    result_matrix = torch.zeros((len(data), len(data)))

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            #print(f"Calculating similarity for pair ({i}, {j})")
            cosine_similarity = torch.nn.functional.cosine_similarity(values[i], values[j], dim=0)
            result_matrix[i, j] = cosine_similarity
            result_matrix[j, i] = cosine_similarity

    return result_matrix
    

def use_embedding_for_sampling_st(dataloader, model, sample_per_dataset, tokenizer, type='prototype_st', save_calibration_distribution=False, model_name="model", dataset_name="dataset"):
    calibration_data = []
    
    for indice, dataset in enumerate(dataloader):
        data_0 = [d[0] for d in dataset]
        st_similarity_matrix =  get_similarity_matrix_with_st(data_0, tokenizer)
        
        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(st_similarity_matrix, dim=1)
        #print(f"Dataset {indice} - Mean cosine similarity: {mean_cosine_similarity.mean().item()}")  # Debugging output
        #print(torch.sort(mean_cosine_similarity, descending=True)[:100])  # Debugging output

        if type == 'prototype_st':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == 'most_different_st':
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)
        
        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])
            
    return calibration_data



def prepare_calibration(model, dataloader, nsamples=128, type='concat', distance='flatten', save_calibration_distribution=False, model_name="model", dataset_name="dataset", count_number_occurrence=False, tokenizer=None):
    """
    Prepare the calibration data by concatenating the datasets and limiting the number of samples.
    """
    sample_per_dataset = nsamples // len(dataloader)
    
    
    if type == 'concat':
        calibration_data = torch.utils.data.ConcatDataset(dataloader)
    if type == 'random_sample':
        calibration_data = random_sample(dataloader, sample_per_dataset)
    elif type == 'prototype' or type == 'most_different': #uses cosine similarity
        calibration_data = use_embedding_for_sampling(dataloader, model, sample_per_dataset, distance, type=type, save_calibration_distribution=save_calibration_distribution, model_name=model_name, dataset_name=dataset_name)
    elif type == 'prototype_iou' or type == 'most_different_iou': #uses intersection over union
        calibration_data = use_embedding_for_sampling_iou(dataloader, model, sample_per_dataset, count_number_occurrence, type=type, save_calibration_distribution=save_calibration_distribution, model_name=model_name, dataset_name=dataset_name)
    elif type == 'prototype_st' or type == 'most_different_st': #uses sentence transformers
        calibration_data = use_embedding_for_sampling_st(dataloader, model, sample_per_dataset, tokenizer, type=type, save_calibration_distribution=save_calibration_distribution, model_name=model_name, dataset_name=dataset_name)

    print(f"Calibration data prepared with {len(calibration_data)} samples.")
    
    return calibration_data   
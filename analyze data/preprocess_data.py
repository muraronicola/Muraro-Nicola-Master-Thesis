import numpy as numpy
import pandas as pd
import numpy as np
import queue
from multiprocessing import Process, Manager
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


base_path = "./data/weight_activation/"
base_name = "{}_{}_{}_{}_{}.npy"
out_path = "./data/processed_data/"

specs_run_name = "specs_run__{}_{}_{}.csv"
file_type_counter = "counter"

model_list = ['Llama-2-7b-hf', 'Mistral-7B-v0.1', 'phi-2', 'Qwen-7B', 'Baichuan-7B'] #['Llama-2-7b-hf', 'Mistral-7B-v0.1', 'phi-2', 'Qwen-7B', 'Baichuan-7B']

dataset_list = ['c4', 'oscar', 'redpajama', 'pile', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 
                'esnli', 'rte', 'boolq', 'commonsense_qa', 'race',
                'winogrande', 'wmt14', 'iwslt', 'opc', 'ds1000', 'mbpp', 'ellipses', 'random']


general_datasets = ['c4', 'oscar', 'redpajama', 'pile']
aritm_reasoning = ['gsm8k', 'svamp', 'mawps']
nlu_inference = ['anli_r1', 'esnli', 'rte']
commonsense_qa = ['boolq', 'commonsense_qa', 'race', 'winogrande']
translation = ['wmt14', 'iwslt']
non_sensical = ['ellipses', 'random']
coding = ['opc', 'ds1000', 'mbpp']
mixed = ['c4', 'gsm8k', 'anli_r1', 'boolq', 'wmt14']


all_categories = [general_datasets, aritm_reasoning, nlu_inference, commonsense_qa, translation, coding, non_sensical]
all_categories_names = ['general_datasets', 'aritm_reasoning', 'nlu_inference', 'commonsense_qa', 'translation', 'coding', 'non_sensical']

all_categories_and_mixed = [general_datasets, aritm_reasoning, nlu_inference, commonsense_qa, translation, coding, non_sensical, mixed]
all_categories_names_and_mixed = ['general_datasets', 'aritm_reasoning', 'nlu_inference', 'commonsense_qa', 'translation',  'coding', 'non_sensical','mixed']


dictionary_id_name = {key: value for key, value in zip(dataset_list, range(len(dataset_list)))}
dictionary_name_id = {key: value for key, value in zip(range(len(dataset_list)), dataset_list)}
dictionary_id_group = {key: value for key, value in zip(range(len(all_categories_names)), all_categories_names)}

runs = range(len(dataset_list)) #range(len(dataset_list))  #range(2) #range(2)
sparsity_vector = [0.1, 0.3, 0.5, 0.7, 0.9] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #range(1) #range(9)
levels = [range(32), range(32), range(32), range(32), range(32)] # range(32)  #range(1) #range(32)
#components = range(1,8)  #range(1) #range(7)




def do_job_intersection(tasks_pending, core_id, filename_file):
    print("Core ID: ", core_id)
    resuls_similarity = pd.DataFrame(columns=['model', 'dataset', 'sparsity', 'level', 'component', 'source_group', 'comparison_group', 'mean_similarity_wrt_comparison_group'])
    
    while True:
        try:
            task = tasks_pending.get_nowait()

            level = task["level"]
            sparsity = task["sparsity"]
            model = task["model"]
            csv_spec_run = task["csv_spec_run"]
            j = task["j"]
            model_path = task["model_path"]
            
            partial_results = get_intersetcion_over_union(level, sparsity, model, csv_spec_run, filename_file, model_path, j)
            resuls_similarity = pd.concat([resuls_similarity, partial_results], ignore_index=True)
            resuls_similarity.to_csv(filename_file, index=False)
            
            print(f"Core ID: {core_id} - Task done: {task['model']} - {task['sparsity']} - {task['level']} - {task['j']}")
        except queue.Empty:
            #print error
            
            print("DONE: " + str(core_id))    
            break
    
    return True


def do_job_counter(tasks_pending, core_id, filename_file):
    print("Core ID: ", core_id)
    result_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements"])

    while True:
        try:
            task = tasks_pending.get_nowait()

            level = task["level"]
            sparsity = task["sparsity"]
            model = task["model"]
            csv_spec_run = task["csv_spec_run"]
            j = task["j"]
            model_path = task["model_path"]
            
            partial_results = get_counter(level, sparsity, model, csv_spec_run, filename_file, model_path, j)
            result_counter = pd.concat([result_counter, partial_results], ignore_index=True)
            result_counter.to_csv(filename_file, index=False)

            print(f"Core ID: {core_id} - Task done: {task['model']} - {task['sparsity']} - {task['level']} - {task['j']}")
            
        except queue.Empty:
            print("DONE: " + str(core_id))    
            break
    
    return True



def do_job_counter_grouped_similarity(tasks_pending, core_id, filename_file):
    print("Core ID: ", core_id)
    result_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements"])

    while True:
        try:
            task = tasks_pending.get_nowait()

            level = task["level"]
            sparsity = task["sparsity"]
            model = task["model"]
            csv_spec_run = task["csv_spec_run"]
            j = task["j"]
            model_path = task["model_path"]
            
            partial_results = get_counter_grouped_similarity(level, sparsity, model, csv_spec_run, filename_file, model_path, j)
            result_counter = pd.concat([result_counter, partial_results], ignore_index=True)
            result_counter.to_csv(filename_file, index=False)

            print(f"Core ID: {core_id} - Task done: {task['model']} - {task['sparsity']} - {task['level']} - {task['j']}")
            
        except queue.Empty:
            print("DONE: " + str(core_id))    
            break
    
    return True



def get_intersetcion_over_union(level, sparsity, model, csv_spec_run, filename_file, model_path, j):
    resuls_similarity = pd.DataFrame(columns=['model', 'dataset', 'sparsity', 'level', 'component', 'source_group', 'comparison_group', 'mean_similarity_wrt_comparison_group'])
    
    components = csv_spec_run['name_id'].unique()
    
    for component in components:
        
        csv_shape = csv_spec_run.loc[(csv_spec_run['layer'] == level) & (csv_spec_run['name_id'] == component)]
        
        shape_0 = csv_shape["shape[0]"].values[0]
        shape_1 = csv_shape["shape[1]"].values[0]
        
        array_data = []
        
        for run in runs:
            dataset = dictionary_name_id[run] #Get the name of the dataset
            path = base_path + model_path + base_name.format(model, sparsity, dataset, level, component)
            #print(f"Processing: {path}")
            packed_data = np.load(path)
            data = np.unpackbits(packed_data, count=shape_0*shape_1).reshape((shape_0, shape_1)).view(bool)
            array_data.append(data)
        
        for run in runs:
            data = array_data[run]
            
            dataset_name = dictionary_name_id[run] #Get the name of the dataset
            
            data_group_id = -1
            for id in range(len(all_categories)):
                if dataset_name in all_categories[id]:
                    data_group_id = id
                    break
            
            group_name = dictionary_id_group[data_group_id]
            
            similarity_for_category = [[] for _ in range(len(all_categories))]
            
            for group in range(len(all_categories)):
                for comparison_dataset in all_categories[group]:
                    
                    if dataset_name != comparison_dataset:
                        data_group = array_data[dictionary_id_name[comparison_dataset]]
                        
                        #Intersection over union
                        intersection = np.logical_and(data_group, data)
                        union = np.logical_or(data_group, data)
                        similarity = np.sum(intersection) / np.sum(union)
                        
                        similarity_for_category[group].append(similarity)
            
            mean_similarity_for_category = [np.mean(similarity_for_category[k]) for k in range(len(similarity_for_category))]
            
            for k in range(len(all_categories)):
                new_data = pd.DataFrame({'model': [model], 'dataset': [dataset_name], 'sparsity': [sparsity], 'level': [level], 'component': [component], 'source_group': [group_name], 'comparison_group': [dictionary_id_group[k]], 'mean_similarity_wrt_comparison_group': [mean_similarity_for_category[k]]})
                resuls_similarity = pd.concat([resuls_similarity, new_data], ignore_index=True)
    
    return resuls_similarity



def get_counter(level, sparsity, model, csv_spec_run, filename_file, model_path, j):
    result_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements"])
    components = csv_spec_run['name_id'].unique()
    
    for component in components:
        
        csv_location = csv_spec_run.loc[(csv_spec_run['layer'] == level) & (csv_spec_run['name_id'] == component)]
        
        shape_0 = csv_location["shape[0]"].values[0]
        shape_1 = csv_location["shape[1]"].values[0]
        
        counter = np.zeros((shape_0, shape_1), dtype=np.int8)
        
        for run in runs:
            dataset = dictionary_name_id[run] #Get the name of the dataset
            
            if dataset in non_sensical:
                continue
                
            path_group = base_path + model_path + base_name.format(model, sparsity, dataset, level, component)
            #print(f"Processing: {path_group}")
            packed_data = np.load(path_group)
            data_group = np.unpackbits(packed_data, count=shape_0*shape_1).reshape((shape_0, shape_1)).view(bool)
            
            counter += data_group.astype(np.int8) #Numero di volte che è stato prunato il neurone
        
        #print(f"counter: {counter}")
        for occurrence in range(len(runs) + 1 - len(non_sensical)):
            mask = (counter == occurrence)
            count = np.sum(mask)
            
            percentage_occurrence = (occurrence / (len(runs)-len(non_sensical))) * 100
            percentage = (count / (shape_0*shape_1))*100
            new_row = pd.DataFrame({"model": [model], "sparsity": [sparsity], "level": [level], "component": [component], "occurrence": [occurrence], "percentage_occurrence": [percentage_occurrence], "number_elements": [count.item()],  "percentage_elements": [percentage.item()]})
            result_counter = pd.concat([result_counter, new_row], ignore_index=True)

    return result_counter





def get_counter_grouped_similarity(level, sparsity, model, csv_spec_run, filename_file, model_path, j):
    result_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements", "category_name"])
    components = csv_spec_run['name_id'].unique()
    
    for component in components:
        
        csv_location = csv_spec_run.loc[(csv_spec_run['layer'] == level) & (csv_spec_run['name_id'] == component)]
        
        shape_0 = csv_location["shape[0]"].values[0]
        shape_1 = csv_location["shape[1]"].values[0]
        
        
        for i in range(len(all_categories_and_mixed)):
            category = all_categories_and_mixed[i]
            category_name = all_categories_names_and_mixed[i]
            
            counter = np.zeros((shape_0, shape_1), dtype=np.int8)
            #print("Category: ", category_name)
            #print("Component category: ", category)
            
            #print(counter[0:10])
            for k in runs:
                dataset = dictionary_name_id[k] #Get the name of the dataset
                if dataset not in category:
                    continue
                
                path_group = base_path + model_path + base_name.format(model, sparsity, dataset, level, component)
                #print(f"Processing: {path_group}")
                packed_data = np.load(path_group)
                data_group = np.unpackbits(packed_data, count=shape_0*shape_1).reshape((shape_0, shape_1)).view(bool)
                
                counter += data_group.astype(np.int8) #Numero di volte che è stato prunato il neurone
            
            #print(counter[0:10])

            for occurrence in range(len(category) + 1):
                mask = (counter == occurrence)
                count = np.sum(mask)
                
                percentage_occurrence = (occurrence / len(category)) * 100
                percentage = (count / (shape_0*shape_1))*100
                new_row = pd.DataFrame({"model": [model], "sparsity": [sparsity], "level": [level], "component": [component], "occurrence": [occurrence], "percentage_occurrence": [percentage_occurrence], "number_elements": [count.item()],  "percentage_elements": [percentage.item()], "category_name": [category_name], "number_members": [len(category)]})
                result_counter = pd.concat([result_counter, new_row], ignore_index=True)

    return result_counter



def main_similarity(num_processors):
    print("-" * 50)
    print("Starting the similarity analysis...")
    print("-" * 50)
    print("\n")
    
    global_resuls_similarity = pd.DataFrame(columns=['model', 'dataset', 'sparsity', 'level', 'component', 'source_group', 'comparison_group', 'mean_similarity_wrt_comparison_group'])

    with Manager() as manager:
        tasks_pending = manager.Queue()
    
        counter = 0
        for i in range(len(model_list)):
            model = model_list[i]
            model_path =  model + "/"
            
            for j in range(len(sparsity_vector)):
                csv_spec_run = pd.read_csv(base_path + model_path + specs_run_name.format(model, sparsity_vector[j], dataset_list[0])) #first run

                for level in levels[i]:
                    parametri  = {"level": level, "j": j, "sparsity": sparsity_vector[j], "model": model, "model_path":model_path, "csv_spec_run": csv_spec_run }
                    tasks_pending.put(parametri)
                    counter += 1

        print("Number of tasks: ", counter)
        
        #create a process
        processes = []
        for core_id in range(num_processors):
            filename_file = out_path + f"/partial/similarity_results_core_{core_id}.csv"
            process = Process(target=do_job_intersection, args=(tasks_pending, core_id, filename_file))
            processes.append(process)
        
        for process in processes:
            process.start()

        for process in processes:
            process.join()
            
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()

    # Merge all the results
    for core_id in range(num_processors):
        filename_file = out_path + f"/partial/similarity_results_core_{core_id}.csv"
        resuls_similarity = pd.read_csv(filename_file)
        global_resuls_similarity = pd.concat([global_resuls_similarity, resuls_similarity], ignore_index=True)

    global_resuls_similarity.to_csv(out_path + "similarity_results.csv", index=False)
    
    print("-" * 50)
    print("Similarity analysis done!")
    print("-" * 50)
    print("\n\n")


def main_counter(num_processors):
    print("-" * 50)
    print("Starting the counter analysis...")
    print("-" * 50)
    print("\n")
    
    global_resuls_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements"])

    with Manager() as manager:
        tasks_pending = manager.Queue()

        counter = 0
        for i in range(len(model_list)):
            model = model_list[i]
            model_path =  model + "/"
            
            for j in range(len(sparsity_vector)):
                csv_spec_run = pd.read_csv(base_path + model_path + specs_run_name.format(model, sparsity_vector[j], dataset_list[0])) #first run
                spars_j = sparsity_vector[j]
                
                for level in levels[i]:
                    parametri  = {"level": level, "j": j, "sparsity": spars_j, "model": model, "model_path":model_path, "csv_spec_run": csv_spec_run }
                    tasks_pending.put(parametri)
                    counter += 1
                    
        print("Number of tasks: ", counter)

        #create a process
        processes = []
        for core_id in range(num_processors):
            filename_file = out_path + f"/partial/counter_core_{core_id}.csv"
            process = Process(target=do_job_counter, args=(tasks_pending, core_id, filename_file))
            processes.append(process)
            
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

    # Merge all the results
    for core_id in range(num_processors):
        filename_file = out_path + f"/partial/counter_core_{core_id}.csv"
        result_counter = pd.read_csv(filename_file)
        global_resuls_counter = pd.concat([global_resuls_counter, result_counter], ignore_index=True)

    global_resuls_counter.to_csv(out_path + "counter.csv", index=False)
    
    print("-" * 50)
    print("Similarity analysis done!")
    print("-" * 50)
    print("\n\n")



def main_counter_grouped_similarity(num_processors):
    print("-" * 50)
    print("Starting the counter_grouped analysis...")
    print("-" * 50)
    print("\n")
    
    global_resuls_counter = pd.DataFrame(columns=["model", "level", "component", "occurrence", "percentage_occurrence", "number_elements", "percentage_elements"])

    with Manager() as manager:
        tasks_pending = manager.Queue()

        counter = 0
        for i in range(len(model_list)):
            model = model_list[i]
            model_path =  model + "/"
            
            for j in range(len(sparsity_vector)):
                csv_spec_run = pd.read_csv(base_path + model_path  + specs_run_name.format(model, sparsity_vector[j], dataset_list[0])) #first run
                spars_j = sparsity_vector[j]
                
                for level in levels[i]:
                    parametri  = {"level": level, "j": j, "sparsity": spars_j, "model": model, "model_path":model_path, "csv_spec_run": csv_spec_run }
                    tasks_pending.put(parametri)
                    counter += 1
                    
        print("Number of tasks: ", counter)

        #create a process
        processes = []
        for core_id in range(num_processors):
            filename_file = out_path + f"/partial/counter_grouped_similarity_core_{core_id}.csv"
            process = Process(target=do_job_counter_grouped_similarity, args=(tasks_pending, core_id, filename_file))
            processes.append(process)
            
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

    # Merge all the results
    for core_id in range(num_processors):
        filename_file = out_path + f"/partial/counter_grouped_similarity_core_{core_id}.csv"
        result_counter = pd.read_csv(filename_file)
        global_resuls_counter = pd.concat([global_resuls_counter, result_counter], ignore_index=True)

    global_resuls_counter.to_csv(out_path + "counter_grouped_similarity.csv", index=False)
    
    print("-" * 50)
    print("Similarity analysis done!")
    print("-" * 50)
    print("\n\n")



if __name__ == '__main__':
    n_cores = 8 #os.cpu_count()
    
    specs_models = pd.DataFrame(columns=['model', 'layer', 'name', 'name_id', 'shape[0]', 'shape[1]'])
    
    for i in range(len(model_list)):
        model = model_list[i]
        
        csv_spec_run = pd.read_csv(base_path + model + "/" + specs_run_name.format(model, sparsity_vector[0], dataset_list[0])) #first run
        out_csv = csv_spec_run[['model', 'layer', 'name', 'name_id', 'shape[0]', 'shape[1]']]
        model_name = out_csv['model'].str.split("/")
        
        #get the first element of the list
        model_name = model_name.str[1]
        
        out_csv.loc[:, 'model'] = model_name

        specs_models = pd.concat([specs_models, out_csv], ignore_index=True)
    
    
    specs_models.to_csv(out_path + "specs_layers.csv", index=False)
    
    main_similarity(n_cores)
    main_counter(n_cores)
    main_counter_grouped_similarity(n_cores)
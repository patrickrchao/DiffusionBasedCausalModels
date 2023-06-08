import numpy as np
import os 
from experiments.structural_equations import get_graph
import pandas as pd 
import networkx as nx

def get_weight_matrices(graph, equations_type, scm_type):
    weights = {}
    # Offset is 1 in nonadditive because we add an extra dimension for noise
    offset = int (equations_type == "nonadditive")
    if scm_type == "random" or scm_type =="ladder":
        for node in graph:
            node_int = int(node[1:])
            super_node = int(np.ceil(node_int/3))
            in_deg = graph.in_degree(node)
            if in_deg > 0:
                if str(super_node)+"_1" not in weights:
                    weights[str(super_node)+"_1"] = np.random.uniform(low=-1, high=1, size=(16 ,in_deg + offset))
                    weights[str(super_node)+"_2"] = np.random.uniform(low=-1, high= 1, size=(3, 16))
    elif scm_type == "sachs":
        for node in graph:
            in_deg = graph.in_degree(node)
            if in_deg > 0:
                if node+"_1" not in weights:
                    weights[node+"_1"] = np.random.uniform(low=-1, high=1, size=(16 ,in_deg + offset))
                    weights[node+"_2"] = np.random.uniform(low=-1, high= 1, size=(1, 16))
    else:
        raise ValueError("SCM type not recognized")
    return weights

def summarize_results(summary_metrics, scaling = 1):
    metrics = ["Obs_MMD", "Int_MMD", "CF_MSE"]
    sorted_cols = sorted(summary_metrics.columns)
    for metric in metrics:
        for col in sorted_cols:
            if metric in col:
                vals = np.array(summary_metrics[col])
                print(f"{col.replace('_',' '):<30} {np.mean(vals)*scaling:>9.4f} \u00B1{np.std(vals)*scaling:>8.4f}")
        print("")

def get_folder(scm_type, equations_type, query_type):
    folder_path = f"experiments/generated_values/{query_type}/{scm_type}_{equations_type}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def get_file_name(seed, n, num_epochs, int_var=None, int_num=None):
    name = f"{seed}_{n}_{num_epochs}"
    if int_var is not None and int_num is not None:
        name = f"{name}_{int_var}_{int_num}.csv"
    else:
        name = f"{name}.csv"
    return name

def initialize_summary_metrics(scm_type,  model_names,num_initializations):
    all_int_var = None
    if scm_type is not "random":
        graph = get_graph(scm_type)
        if scm_type == "ladder":
            all_int_var = sorted(["x01","x04"])
        else:
            all_int_var = sorted([node for node in graph if len(nx.descendants(graph,node)) > 0])
    
    summary_names = [model + "_Obs_MMD" for model in model_names ]
    # If using a random scm, then the column names are inconsistent
    if scm_type is not "random":
        summary_names.extend([model + "_" + metric + "_" + int_var for metric in ["Int_MMD", "CF_MSE"] for model in model_names for int_var in all_int_var])
    else:
        summary_names.extend([model + "_" + metric + "_" + str(ind) for metric in ["Int_MMD", "CF_MSE"] for model in model_names for ind in range(1,4)])
    
    summary_metrics = pd.DataFrame(np.zeros((num_initializations, len(summary_names))), columns=summary_names)
    return summary_metrics, all_int_var

def reindex_columns(column_order, dfs):
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    result = [df.reindex(columns = column_order) for df in dfs]
    if len(result) == 1:
        return result[0]
    else:
        return result
        

from dowhy.gcm import (
                       InvertibleStructuralCausalModel, 
                       counterfactual_samples, 
                       interventional_samples, 
                       draw_samples,
                       is_root_node)
import dowhy.gcm as cy
import networkx as nx, numpy as np, pandas as pd
from experiments.data_generation import ExperimentationModel
from experiments.structural_equations import *
from experiments.MMD import mmd
import sys

from experiments.exp_helper import *

import warnings
warnings.filterwarnings('ignore')



def get_data_loader(dataset_params, intervention, n_samples = 100):
    data_module = create_data_from_data_params(dataset_params, new_data=True, n= 2 * n_samples)
    data_module.batch_size = n_samples
    data_loader = data_module.test_dataloader()
    if intervention is not None:
        data_loader.dataset.set_intervention(intervention)
        data_loader = data_module.test_dataloader()
    return data_loader


def convert_data_to_pandas(var_to_idx, *all_data):
    def converting(data):
        if not isinstance(data, np.ndarray):
            data = data.numpy()
        df = pd.DataFrame(data)
        df.columns = columns
        return df
    columns = [None for _ in range(len(var_to_idx))]
    for var in var_to_idx:
        columns[var_to_idx[var]] = var

    result = [converting(x) for x in all_data]
    if len(result) == 1:
        return result[0]
    else:
        return result
    
def get_train_data(data_module, graph, true_model):
    data_loader = data_module.train_dataloader()
    train_u = convert_data_to_pandas(data_loader.dataset.var_to_idx, data_loader.dataset.U)
    train_factual = true_model.data_from_noise(train_u)
    return train_factual

def obs_helper(models, true_model, scm_type, equations_type, seed, graph, column_order,dataset_params):
    nonroot_nodes = [node for node in sorted(graph.nodes) if not is_root_node(graph, node)]
    all_model_names = list(models.keys())
    obs_mmds = {}
    obs_folder = get_folder(scm_type, equations_type, "observational")
    all_col_sub = lambda names: [x+"_"+col for x in names for col in column_order]
    
    true_obs = reindex_columns(column_order,draw_samples(true_model, num_samples = num_obs_samples))
    obs_preds = pd.DataFrame(columns=all_col_sub(["Truth"] + all_model_names))
    obs_preds[all_col_sub(["Truth"])] = true_obs
    if use_vaca:
        obs_data_loader = get_data_loader(dataset_params, None, num_obs_samples)
    
    for model_name in models:
        model = models[model_name]
        if model_name in ["VACA", "CAREFL"]:
            input_params= {"data_loader": obs_data_loader,
                          "normalize" :False}
            if model_name == "VACA":
                input_params["num_batches"] = 1
            _, cur_obs_pred, _ = model.get_observational_distr(**input_params)
            cur_obs_pred = convert_data_to_pandas(obs_data_loader.dataset.var_to_idx, cur_obs_pred)  
        else:
            cur_obs_pred = reindex_columns(column_order,draw_samples(model, num_samples = num_obs_samples))

        obs_preds[all_col_sub([model_name])] = cur_obs_pred   
        obs_mmds[model_name] = mmd(cur_obs_pred[nonroot_nodes], true_obs[nonroot_nodes])
        print(f"Obs MMD {model_name}: {obs_mmds[model_name]:>.4f}")
    print("")
    file_name = get_file_name(seed, n, num_epochs)
    if save_preds:
        obs_preds.to_csv(f"{obs_folder}/{file_name}", index=False)
    return obs_mmds
        
def exp_helper(models, int_var, true_model, scm_type, equations_type, seed, graph, column_order=None,dataset_params=None):
    """Computes observational, intervention, and counterfactual values for dictionary of models
    Stores all values in pandas data frame and saves to experiments/generated_values folder
    """
    temp_data = draw_samples(true_model, 5000)
    lower_quantile, upper_quantile = np.quantile(temp_data[int_var],[0.1,0.9])
    all_int_val = np.linspace(lower_quantile, upper_quantile, num = num_interventions)

    descendants = sorted(list(nx.descendants(graph, int_var)))
    
    all_model_names = list(models.keys())
    cf_mses, int_mmds = {}, {}
    for model_name in models:
        cf_mses[model_name] = []
        int_mmds[model_name] = []
    int_folder = get_folder(scm_type, equations_type, "interventional")
    cf_folder = get_folder(scm_type, equations_type, "counterfactual")
    
    desc_col_sub = lambda names: [x+"_"+desc for x in names for desc in descendants]

    for int_num, int_val in enumerate(all_int_val):
        cf_preds = pd.DataFrame(columns = desc_col_sub(["Truth", "Factual"] + all_model_names))
        int_preds = pd.DataFrame(columns = desc_col_sub(["Truth"] + all_model_names))
        
        vaca_intervention = {int_var : int_val}
        intervention = {int_var : lambda x : int_val}

        if use_vaca:
            data_loader = get_data_loader(dataset_params, vaca_intervention, num_int_samples)            
            # Cannot use factual values from VACA, instead need to generate our own from noise
            factual_u = convert_data_to_pandas(data_loader.dataset.var_to_idx, data_loader.dataset.U)
            factual = reindex_columns(column_order,true_model.data_from_noise(factual_u))
        else:
            factual, factual_u = reindex_columns(column_order,true_model.sample(num_int_samples))
        
        gt_cf_full = reindex_columns(column_order,true_model.get_counterfactuals(intervention, factual_u))
        gt_cf = gt_cf_full[descendants]
        true_int = reindex_columns(column_order,interventional_samples(true_model, 
                                           intervention, 
                                           num_samples_to_draw = num_int_samples))
                                   
        true_int = true_int[descendants]
        
        for model_name in models:
            model = models[model_name]

            # Get model interventional and counterfactual predictions
            if model_name in ["VACA", "CAREFL"]:
                input_params = {"data_loader": data_loader,
                                    "x_I": vaca_intervention,
                                    "num_batches": 1,
                                    "normalize": False}
                cur_cf_pred, vaca_gt_cf_full, _ = convert_data_to_pandas(data_loader.dataset.var_to_idx,                               
                        *[x['all'] for x in model.get_counterfactual_distr(**input_params)])
                vaca_gt_cf = vaca_gt_cf_full[descendants]
                assert np.max(np.abs(gt_cf - vaca_gt_cf))[0] < 1e-4, f"Inconsistent counterfactual values from {model_name} and DCM"
 
                if model_name == "CAREFL":
                    input_params.pop("num_batches")
                cur_int_pred, _ = convert_data_to_pandas(data_loader.dataset.var_to_idx,                                   
                                *[x['all'] for x in model.get_interventional_distr(**input_params)])
            else:
                cur_cf_pred = reindex_columns(column_order,counterfactual_samples(model, intervention, observed_data = factual))
                cur_int_pred = reindex_columns(column_order,interventional_samples(model, intervention, num_samples_to_draw = num_int_samples))
            cur_cf_pred = cur_cf_pred[descendants]
            cur_int_pred = cur_int_pred[descendants]
            
            # Save predictions, Int. MMD and CF MSE
            cf_preds[desc_col_sub([model_name])] = cur_cf_pred
            int_preds[desc_col_sub([model_name])] = cur_int_pred

            cf_mses[model_name].append(np.mean(np.square(cur_cf_pred - gt_cf))[0])
            int_mmds[model_name].append(mmd(cur_int_pred, true_int))
            
        cf_preds[desc_col_sub(["Truth"])] = gt_cf
        cf_preds[desc_col_sub(["Factual"])] = factual[descendants]        
        int_preds[desc_col_sub(["Truth"])] = true_int
        
        cf_preds[f"Intervene {int_var}"] = int_val
        int_preds[f"Intervene {int_var}"] = int_val
        
        file_name = get_file_name(seed, n, num_epochs, int_var, int_num)
        if save_preds:
            cf_preds.to_csv(f"{cf_folder}/{file_name}", index=False)
            int_preds.to_csv(f"{int_folder}/{file_name}", index=False)
    
    
    for model_name in models:
        int_mmds[model_name] = np.mean(np.array(int_mmds[model_name]))
        cf_mses[model_name] = np.mean(np.array(cf_mses[model_name]))
        print(f"{int_var} Int MMD {model_name}: {int_mmds[model_name]:>.4f}")
    print("")
    for model_name in models:
        print(f"{int_var} CF MSE {model_name}: {cf_mses[model_name]:>.4f}")
    print("")   
    return int_mmds, cf_mses

def all_exp(equations_type, scm_type, use_vaca): 

    model_names = ["DCM","ANM"]
    if use_vaca:
        model_names.extend([ "VACA", "CAREFL"])

    summary_metrics, all_int_var = initialize_summary_metrics(scm_type, model_names, num_initializations)
    
    dataset_params, true_model = None, None
    for i in range(num_initializations):
        seed = i
        np.random.seed(seed)

        graph = get_graph(scm_type)

        if len(graph.nodes) >= 10:
            # Need this for how VACA sorts nodes
            column_order = ['x{:02}'.format(i+1) for i in range(len(graph.nodes()))]
        else:
            column_order = ['x'+str(i+1) for i in range(len(graph.nodes()))]
        if scm_type == "random" or scm_type == "sachs":
            valid_var = [node for node in graph if len(nx.descendants(graph,node)) > 0]
            if len(valid_var) < 3:
                all_int_var = sorted(valid_var)
            else:
                all_int_var = sorted(np.random.choice(valid_var,3, replace = False))
        weights = None
        if scm_type in ["random", "ladder", "sachs"]:
            weights = get_weight_matrices(graph, equations_type,scm_type)
        
        structural_equations, noise_distributions = select_struct_and_noise(equations_type, scm_type, weights, graph)
        

        true_model = ExperimentationModel(graph.copy(),scm_type, structural_equations, noise_distributions)

        print(f"Initialization {i+1} out of {num_initializations}\n----------------------")
        diff_model = create_diff_model(scm_type, params, graph.copy())
        causal_model = InvertibleStructuralCausalModel(graph.copy())
        
        if use_vaca:
            vaca_data_module, cfg, dataset_params = create_data(n,seed = seed,
                                                                structural_equations = structural_equations,
                                                                noise_distributions = noise_distributions,
                                                                graph = graph.copy(),
                                                                name = scm_type,
                                                                model_type = "VACA",
                                                                equations_type = equations_type)
            _, carefl_cfg, dataset_params = create_data(n,seed = seed,
                                                                structural_equations = structural_equations,
                                                                noise_distributions = noise_distributions,
                                                                graph = graph.copy(),
                                                                name = scm_type,
                                                                model_type = "CAREFL",
                                                                equations_type = equations_type)
            vaca_model = create_model(cfg, vaca_data_module)    
            carefl_model = create_model(carefl_cfg, vaca_data_module)
            train_df = get_train_data(vaca_data_module, graph, true_model)

            
            fit_model(vaca_model, cfg, vaca_data_module)
            fit_model(carefl_model, carefl_cfg, vaca_data_module)
        else:
            train_df, _ = true_model.sample(n)
            train_df = reindex_columns(column_order,train_df)
        
        cy.fit(diff_model, train_df)
        # try:
        #     cy.auto.assign_causal_mechanisms(causal_model,
        #                                  train_df,
        #                                  quality=cy.auto.AssignmentQuality.BETTER,
        #                                  override_models=True)
        # except:
        
        cy.auto.assign_causal_mechanisms(causal_model,
                                         train_df, override_models = True)
        cy.fit(causal_model, train_df)
         
        models = {
                  model_names[0]:diff_model, 
                  'ANM': causal_model}
        if use_vaca:
            models['VACA'] = vaca_model
            models['CAREFL'] = carefl_model
            
        obs_mmds = obs_helper(models, true_model, scm_type, equations_type, seed, graph, column_order,dataset_params)
        for model_name in model_names:
            summary_metrics.loc[i, model_name+"_Obs_MMD"] = obs_mmds[model_name]
        
        for j,int_var in enumerate(all_int_var):
            int_mmds, cf_mses = exp_helper(models, int_var, true_model, scm_type, equations_type, seed, graph, column_order,dataset_params)
            for model_name in model_names:
                
                int_name = model_name+"_Int_MMD"+"_"
                cf_name = model_name+"_CF_MSE"+"_"
                if scm_type != "random":
                    int_name += int_var
                    cf_name += int_var
                else:
                    int_name += str(j+1)
                    cf_name += str(j+1)
                summary_metrics.loc[i, int_name] = int_mmds[model_name]
                summary_metrics.loc[i, cf_name] = cf_mses[model_name]
        summary_metrics.to_csv(f"experiments/metric_summaries/{scm_type}_{equations_type}_{n}_{num_epochs}.csv", index=False)
    return summary_metrics


if __name__ == '__main__':
    # These parameters may be changed
    n = 5000
    num_epochs = 500
    num_initializations = 10 
    num_int_samples = 100
    num_interventions = 20
    num_obs_samples = 1000
    use_vaca = True
    if use_vaca:
        warnings.warn("To run VACA and CAREFL experiments, please install VACA from: https://github.com/psanch21/VACA")
        sys.path.append('VACA_modified/')
        from VACA_helper import *
    save_preds = True
    params = {'num_epochs' : num_epochs,
              'lr' : 1e-4,
              'batch_size': 64,
              'hidden_dim' : 64,
              'use_positional_encoding': False,
              'weight_decay': 0,
              'lambda_loss' : 0,
              'clip': False,
              'verbose' : False}

    scm_types = ["triangle","chain","Y","diamond","ladder","random","sachs"]
    equations_types = ["nonlinear","nonadditive"]
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        scm_list = []
        equations_list = []
        print(arg)
        for scm in scm_types:
            if scm.lower() in arg.lower():
                scm_list.append(scm)
        for equation in equations_types:
            if equation.lower() in arg.lower():
                equations_list.append(equation)
        if len(scm_list) == 0 or len(equations_list) == 0:
            print("Invalid arg")
            raise Exception
        
    else:                        
        scm_list = scm_types #["sachs", "chain","diamond","Y", "diamond", "triangle", "chain"]
        equations_list = equations_types#["nonlinear","nonadditive"]
    all_summary_metrics = {}
    print(f"SCMs: {scm_list}")
    print(f"Equations: {equations_list}")

    for equations_type in equations_list:
        for scm_type in scm_list:
            print(f"\nEvaluating {scm_type} {equations_type} Data with n={n} and {num_epochs} epochs for DCM\n-------------------------------\n")
            summary_metrics = all_exp(equations_type, scm_type, 
                          use_vaca=use_vaca)

            scaling = 100
            print(f"\nSummary for {scm_type} {equations_type} Data with n={n} and {num_epochs} epochs when multiplied by {scaling}\n-----------------------------------")
            summarize_results(summary_metrics, scaling)
            all_summary_metrics[scm_type+" "+equations_type] = summary_metrics
            
    for equations_type in equations_list:
        for scm_type in scm_list:
            summary_metrics =  all_summary_metrics[scm_type+" "+equations_type] 
            print(f"\nSummary for {scm_type} {equations_type} Data with n={n} and {num_epochs} epochs when multiplied by {scaling}\n-----------------------------------")
            summarize_results(summary_metrics, scaling)
            


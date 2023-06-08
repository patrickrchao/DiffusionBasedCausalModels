import numpy as np
import sys
sys.path.append('../')
from dowhy.gcm import ScipyDistribution, InvertibleStructuralCausalModel, EmpiricalDistribution

from scipy.stats import norm, multivariate_normal
import networkx as nx
from model.diffusion import CausalDiffusionModel, create_model_from_graph

import math



sqrt = np.sqrt
Normal = lambda mean, variance: ScipyDistribution(norm, loc=mean, scale = sqrt(variance))

def select_struct_and_noise(equations_type, scm_type, weights = None, graph=None):
    equations_type = equations_type.lower()
    scm_type = scm_type.lower()
    if scm_type == "random":
        if graph is None:
            raise ValueError("Must first sample graph from `get_graph` when using random scm")
    else:
        graph = get_graph(scm_type)
    noises_distr = {}
    if scm_type == "sachs":
        mapping = {"x01":"plcg",
        "x02":"PIP3",
        "x03":"PIP2" ,
        "x04":"PKC" ,
        "x05":"PKA" ,
        "x06":"pjnk" ,
        "x07":"P38" ,
        "x08":"praf" ,
        "x09":"pmek",
        "x10":"p44/42",
        "x11":"pakts473"}
        from cdt.data import load_dataset
        data_sachs, _ = load_dataset("sachs")
        def sachs_empirical(node):
            sachs_node = mapping[node]
            dist = EmpiricalDistribution()
            empirical_data = data_sachs[sachs_node].values
            empirical_data = (empirical_data - np.min(empirical_data))/(np.max(empirical_data) - np.min(empirical_data)) * 2 - 1
            dist.fit(empirical_data)
            return dist

    for node in graph.nodes:
        if scm_type != "sachs":
            noises_distr[node] = Normal(0, 1)
        else:
            if len(graph.in_edges(node)) > 0:
                noises_distr[node] = Normal(0, 1)
            else:
                noises_distr[node] = sachs_empirical(node)


    if scm_type == "triangle":
        if equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (u2 + 2 * x1 ** 2) / 3,
                'x3': lambda u3, x1, x2: (u3 + 20 / (1 + np.exp(-x2 ** 2 + x1))) / 2.41
            }

        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (x1/((u2 + x1)**2+1) + u2/4) / 0.47,
                'x3': lambda u3, x1, x2: ((np.abs(u3)+0.3) * (-x1 + x2 / 2 + np.abs(u3)/5)**2) /0.93 
            }

    elif scm_type == "chain":
        if equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (np.exp(x1 / 2) + u2 / 4) / 0.65, 
                'x3': lambda u3, x2: ((x2 - 5) ** 3 / 15 + u3) / 2.13,
            }
        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: 1/((u2+x1)**2 + 0.5) / 0.63,
                'x3': lambda u3, x2: np.sqrt(x2 + np.abs(u3)) / (0.1 + x2)
            }

    elif scm_type == "diamond":
        if equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (x1 ** 2 + u2 / 2) / 1.5,
                'x3': lambda u3, x1, x2: (x2 ** 2 - 2 / (1 + np.exp(-x1)) + u3 / 2) / 1.27,
                'x4': lambda u4, x2, x3: (x3 / (np.abs(x2 +  x3) + 0.5) + u4 / 5) / 0.54,
            }

        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (np.sqrt(np.abs(x1)) * (np.abs(u2)+0.1)/2 + np.abs(x1) + u2/5)/0.83,
                'x3': lambda u3, x1, x2: 6 / (1 + (np.abs(u3)+0.5) * np.exp(-x2  + x1)),
                'x4': lambda u4, x2, x3: (((x3 + x2 + u4/4-7)**2) - 20)/ 7.27
            }

    elif scm_type == "y":
        if equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1, x2: (- x2**2 + 4/(1+np.exp(-x1 - x2)) +  u3/2) / 1.83,
                'x4': lambda u4, x3: (20/(1+np.exp(x3**2/2-x3)) +  u4) / 3.26,
            }

        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1, x2: (x1 - 2*x2 - 2) * (np.abs(u3) + 0.2) / 2.87,
                'x4': lambda u4, x3: 1/(np.abs(u4+x3*2) + 0.5) / 0.43
            }

    elif scm_type == "hexagon":
        if equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1 : x1 ** 2 / 5  - x1 + u2,
                'x3': lambda u3, x1: x1**2 / 5 - 4/(1+np.exp(-x1)) +  u3,
                'x4': lambda u4, x2, x3: (x3**2 +  x2**2)/5 +  u4,
                'x5': lambda u5, x2, x3: (x3**2 +  x2**2)/5 +  u5,
                'x6': lambda u6, x4, x5: (x4**2 +  x5**2)/5 +  u6,
            }

        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: x1 ** 2 / 5  - x1 + u2,
                'x3': lambda u3, x1: x1**2 / 5 - 4/(1+np.exp(-x1)) +  u3,
                'x4': lambda u4, x2, x3: (x3**2 +  x2**2)/5 +  u4,
                'x5': lambda u5, x2, x3: (x3**2 +  x2**2)/5 +  u5,
                'x6': lambda u6, x4, x5: (x4**2 +  x5**2)/5 +  u6,
            }


    elif scm_type == "bivariate":
        if equations_type == "exp":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: np.exp(x1) + u2,
            }
        elif equations_type == "linear":
            structural_eq = {
            'x1': lambda u1: u1+10,
            'x2': lambda u2, x1: 2*x1 + u2+10,
            }
        elif equations_type == "nonlinear":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: x1**2 + u2,
            }
        elif equations_type == "nonlinear2":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1: (20/(1+np.exp(x1**2/2-x1)) +  u2) / 3.26,
            }
        elif equations_type == "nonadditive":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1:  4/(1+2*np.exp(-x1)) + x1*u2*4
            }
        elif equations_type == "sign":
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2, x1:  x1 + (2*np.sign(u2)-1)*x1 
            }
    elif scm_type == "ladder" or scm_type == "random":

        structural_eq = {}
        
        for node in graph.nodes:
            ind = int (node[1:])
            super_node = int (math.ceil(ind/3))
            if graph.in_degree(node) > 0:
               
                w1, w2 = weights[str(super_node)+"_1"], weights[str(super_node)+"_2"]
                node_subind = (ind-1) % 3
                
                if equations_type == "nonadditive":
                    def create_helper(mat1,mat2, cur_ind):
                        def helper(*params):
                            full_params = np.concatenate(params,axis=1).T
    
                            return (mat2 @ silu(mat1 @ full_params) * 0.75)[cur_ind, :]
                        return helper  
                else:
                    def create_helper(mat1,mat2, cur_ind):
                        def helper(u,*params):
                            full_params = np.concatenate(params,axis=1).T
                            return (mat2 @ silu(mat1 @ full_params) * 0.75)[cur_ind, :] + u.reshape(-1)
                        return helper 

                structural_eq[node] = create_helper(w1, w2,node_subind)

            else:
                structural_eq[node] = lambda u: u 
    elif scm_type == "sachs":

        structural_eq = {}
        
        for node in graph.nodes:
            
            
            if graph.in_degree(node) > 0:
               
                w1, w2 = weights[node+"_1"], weights[node+"_2"]
                
                
                if equations_type == "nonadditive":
                    def create_helper(mat1,mat2):
                        def helper(*params):
                            full_params = np.concatenate(params,axis=1).T
                            return (mat2 @ silu(mat1 @ full_params) * 0.75)[0, :]
                        return helper  
                else:
                    def create_helper(mat1,mat2):
                        def helper(u,*params):
                            full_params = np.concatenate(params,axis=1).T
                            return (mat2 @ silu(mat1 @ full_params) * 0.75)[0, :] + u.reshape(-1)
                        return helper 
                structural_eq[node] = create_helper(w1, w2)
            else:
                structural_eq[node] = lambda u: u 

    else:
        raise NotImplementedError
    return structural_eq, noises_distr

def get_graph(scm_type):
    scm_type = scm_type.lower()
    if scm_type == "triangle":
        g = nx.DiGraph([('x1', 'x2'),('x2','x3'), ('x1', 'x3')])
    elif scm_type == "chain":
        g = nx.DiGraph([('x1', 'x2'),('x2','x3')])
    elif scm_type == "diamond":
        g = nx.DiGraph([('x1', 'x2'),('x2','x3'), ('x1', 'x3'), ('x2', 'x4'),  ('x3', 'x4') ])
    elif scm_type == "y":
        g = nx.DiGraph([('x1', 'x3'),('x2', 'x3'), ('x3', 'x4')])
    elif scm_type == "bivariate":
        g = nx.DiGraph([('x1', 'x2')])
    elif scm_type == "ladder":
        edge_list = []
        super_node_edge_list = [('x1','x2'), ('x1', 'x3'), ('x2', 'x4'),
                                ('x3','x4'), ('x3', 'x5'), ('x4', 'x6'),
                                ('x5','x6'), ('x5', 'x7'), ('x6', 'x8'),
                                ('x7','x8'), ('x7', 'x9'), ('x8', 'x10'),
                                ('x9', 'x10')]
        for edge in super_node_edge_list:
            edge_list.extend(add_multi_edges(edge[0],edge[1]))
           
        g = nx.DiGraph(edge_list)
    elif scm_type == "random":
        g = random_dag(10, 0.3)  
    elif scm_type == "sachs":
        # from cdt.data import load_dataset
        # data_sachs, g = load_dataset("sachs")
        # g = nx.DiGraph([('Plcg', 'PIP3'),('PLcg','PIP2'), ('PIP3', 'PIP2'), 
        # ('PKC', 'PKA'),  ('PKC', 'Jnk'), ('PKC', 'P38'),  ('PKC', 'Raf'), ('PKC', 'Mek'),  
        # ('PKA', 'Erk'), ('PKA', 'Jnk'),  ('PKA', 'Raf'), ('PKA', 'Mek'),  ('PKA', 'Erk'), ('PKA', 'Akt'), ('PKA', 'P38'),
        # ('Raf', 'Mek'),('Mek', 'Erk'),('Erk', 'Akt')  ])
        # g = nx.DiGraph([('plcg', 'PIP3'),('plcg','PIP2'), ('PIP3', 'PIP2'), 
        # ('PKC', 'PKA'),  ('PKC', 'pjnk'), ('PKC', 'P38'),  ('PKC', 'praf'), ('PKC', 'pmek'),  
        # ('PKA', 'p44/42'), ('PKA', 'pjnk'),  ('PKA', 'praf'), ('PKA', 'pmek'),  ('PKA', 'p44/42'), ('PKA', 'pakts473'), ('PKA', 'P38'),
        # ('praf', 'pmek'),('pmek', 'p44/42'),('p44/42', 'pakts473')  ])
        g = nx.DiGraph([('x01', 'x02'),('x01','x03'), ('x02', 'x03'), 
        ('x04', 'x05'),  ('x04', 'x06'), ('x04', 'x07'),  ('x04', 'x08'), ('x04', 'x09'),  
        ('x05', 'x10'), ('x05', 'x06'),  ('x05', 'x08'), ('x05', 'x09'),  ('x05', 'x10'), ('x05', 'x11'), ('x05', 'x07'),
        ('x08', 'x09'),('x09', 'x10'),('x10', 'x11')  ])
    else:
        print(f"SCM type {scm_type} not recognized.")
        raise NotImplementedError
    return g


def create_diff_model(scm_type, params, g=None):
    if scm_type == "random":
        if g is None:
            raise ValueError("Graph must be provided for random scm")
    else:
        g = get_graph(scm_type)
    model = create_model_from_graph(g, params)
    return model

def silu(x):
    sigmoid = np.where(x >= 0,  1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return np.multiply(x, sigmoid)

def random_dag(num_super_nodes, p = 0.3):
    
    connected = False
    num_tries = 0
    while not connected and num_tries < 50:
        num_tries += 1
        adj_mat = np.zeros((num_super_nodes,num_super_nodes))
       # Get the strictly upper triangular indices
        for i in range(num_super_nodes):
            for j in range(i + 1, num_super_nodes):
                if np.random.rand() < p:
                    adj_mat[i, j] = 1

        g = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
        connected = nx.is_weakly_connected(g)
    if num_tries > 20:
        raise ValueError("Could not create a connected graph")
    edges= []
    for (i,j) in g.edges():
        edges.extend(add_multi_edges(i+1,j+1))
    
    graph =  nx.DiGraph(edges)
    
    return graph

def add_multi_edges(super_out_node, super_in_node):

    if type(super_out_node) == str:
        super_out_node = int (super_out_node[1:])
        super_in_node = int (super_in_node[1:])
        
    new_edges = []
    for i in range(1,4):
        for j in range(1,4):
            # This is only valid for 99 total nodes
            node1 = str.zfill(str((super_out_node - 1) * 3 + i), 2)
            node2 = str.zfill(str((super_in_node - 1) * 3 + j), 2)
            new_edges.append(("x" + node1, "x" + node2))
    return new_edges

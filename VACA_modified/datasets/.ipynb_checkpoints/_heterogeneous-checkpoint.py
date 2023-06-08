import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data.data import Data

from datasets._adjacency import Adjacency
from datasets.transforms import ToTensor
from utils._errors import IsHeterogeneousError
from utils.distributions import *

flatten = lambda t: [item for sublist in t for item in sublist]
import utils.likelihoods as ul
from utils.constants import Cte

from utils.args_parser import list_substract

from typing import List, Any, Dict


# %%
class HeterogeneousSCM(torch.utils.data.Dataset):

    def __init__(self, root_dir: str,
                 transform: Any,
                 nodes_to_intervene: List[str],
                 nodes_list: List[str],
                 adj_edges: Dict[str, list],
                 structural_eq: Dict[str, Any],
                 noises_distr: Dict[str, Any],
                 lambda_: float):

        """
        Base class for the SCM based datasets.
        Args:
            root_dir:
            transform:
            nodes_to_intervene:
                The list of strings with the identifier of the nodes in which we would like to intervene. E.g., ['a', 'b']
            nodes_list:
                The list of strings with the identifier for each node. E.g., ['a', 'b']
            adj_edges:
                Dictionary of edges. Keys are the parents and values are list of children. E.g., {'a': ['b'], 'b': []}.
            structural_eq:
                Dictionary of functions. Keys are nodes and values are the function representing the strcutural equation.
                If true SCM is unknown this parameter should be None
            noises_distr:
                Dictionary of noise distributions. Keys are nodes and values are the function representing the noise distributions.
                If true SCM is unknown this parameter should be None
            lambda_:
                The parameter for the DeltaLikelihood.
        """

        assert lambda_ > 0.0, 'Lambda should be a positive real number!'

        self.root_dir = root_dir
        self.transform = transform

        self.nodes_list = nodes_list  # List of nodes. E.g., ['a', 'b']
        self.adj_edges = adj_edges  # Dictionary of edges. E.g., {'a': ['b'], 'b': []}
        self.structural_eq = structural_eq
        self.noises_distr = noises_distr

        self.num_nodes = len(nodes_list)

        self.lambda_ = lambda_

        self.has_ground_truth = isinstance(structural_eq, dict) and isinstance(noises_distr, dict)

        # X represent the featues without the zeros
        self.X = None
        self.U = None  # The exogenous variables are only available when has_ground_truth=True

        # X0 represents the features with filling with zeros
        self.X0, self.mask_X0 = None, None
        self.dim_of_x_in_x0 = None
        self.total_num_dim_x0 = None

        # Intervention variables
        self.x_I = None  # Set variables intervened
        self.I_noise = False

        self.adj_object = None
        self.nodes_to_intervene = nodes_to_intervene

    @property
    def node_dim(self):
        """
        This should raise an error whenever nodes have different dimensions
        Returns: integer with the number of dimensions of each node

        """
        node_dim_list = self.get_node_dimensions()
        node_dim_list = list(set(node_dim_list))

        if len(node_dim_list) > 1:
            raise IsHeterogeneousError
        else:
            return node_dim_list[0]

    @property
    def node_per_dimension_list(self):
        """
        This is a List of intergers. Element i contains the node_id of the i-th column in self.X
        Returns:
            List of ints.
        """

        node_dim_list = self.get_node_dimensions()
        output = []
        for i, dim in enumerate(node_dim_list):
            output.extend([self.nodes_list[i], ] * dim)

        return output

    @property
    def likelihoods(self):
        """
        List of lists.
            i-th element of the outer list contains a List of likelihood for the i-th node
            j-th element of the inner list contains the likelihood for the j-th (group of) dimension(s) of the node.
        Returns:
            List of lists with Likelihood objects
        """
        raise NotImplementedError

    @property
    def std_list(self):
        """
        Values (proportional to the standard deviation) to be intervened-on.
        E.g.,  X_intervened = X_mean + std_list[0]
        Returns:
            List of ints
        """ #[-1, -0.5, -0.1, 0.1, 0.5, 1]
        return [-1, 1]

    @property
    def var_to_idx(self):
        """
        Mapping from node_name to index
        Returns:
            Dict[str, int]
        """
        return {node: i for i, node in enumerate(self.nodes_list)}

    @property
    def is_heterogeneous(self):
        """
        Flag to check if the SCM heterogeneous. An SCM is heterogeneous if
            - Nodes have different dimensions
            - Nodes have different likelihoods
        Returns:

        """
        node_dim_list = self.get_node_dimensions()
        node_dim_list = list(set(node_dim_list))

        if len(node_dim_list) > 1:
            return True

        likelihood_names = []
        for lik in flatten(self.likelihoods):
            likelihood_names.append(lik.name)
        likelihood_names = list(set(likelihood_names))

        if len(likelihood_names) > 1:
            return True
        else:
            return False

    @property
    def likelihood_list(self):
        """
        if the SCM is heterogeneous return likelihoods
        if the SCM is not heterogeneous return the likelihood object (which is shared by all the nodes)
        Returns:
            List[List[Likelihoods]]
            or
            Likelihood
        """
        if self.is_heterogeneous:

            return self.likelihoods
        else:
            return self.likelihoods[0][0]

    @property
    def largest_node_dim(self):
        """
        The largest dimension among the dimensions of the nodes in the SCM.
        Returns:

        """
        return max(self.get_node_dimensions())

    @property
    def num_edges(self):
        return self.adj_object.num_edges

    @property
    def num_samples(self):
        return self.X.shape[0]

    @property
    def num_dimensions(self):
        """
        Total number of dimensions in the SCM. E.g., this variable is num_nodes if the SCM is homogeneous with unidimensional nodes.
        Returns:
            int
        """
        return sum(self.get_node_dimensions())

    @property
    def num_parameters(self):
        """
        The total number of likelihood parameters in the SCM.
        Returns:

        """
        likelihoods = flatten(self.likelihoods)
        return sum([lik.params_size for lik in likelihoods])

    # %% Methods

    def is_toy(self):
        return True

    def _get_lik(self, lik_str: str,
                 dim: int,
                 normalize):
        """
        Likelihood object according given a name and dimensions. Also, we can specify if we want to normalize this likelihood
        Args:
            lik_str: Name (abbreviated) of the distribution.
            dim: dimension of the distribution
            normalize: normalization mode
        Returns:
            BaseLikelihood object
        """
        if lik_str == 'd':
            return ul.DeltaLikelihood(dim, lambda_=self.lambda_, normalize=normalize)
        elif lik_str == 'cb':
            return ul.ContinousBernoulliLikelihood(dim, normalize=normalize)
        elif lik_str == 'b':
            return ul.BernoulliLikelihood(dim, normalize=normalize)
        elif lik_str == 'c':
            return ul.CategoricalLikelihood(dim, normalize=normalize)
        else:
            raise NotImplementedError

    def _get_G(self):
        """
        Convert the adjacency matrix into a networkx Directed graph
        Returns:

        """
        adj = self.dag.copy()

        np.fill_diagonal(adj, 0.0)
        G = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph)
        return G

    def _get_x_from_x0(self, x0):
        """
        Convert the extended samples matrix x0 into the samples matrix x. In other words, remove the redundant columns containing 0s.
        Args:
            x0:

        Returns:

        """
        return x0[:, flatten(self.dim_of_x_in_x0)]

    def get_topological_nodes_pa(self):
        '''
        Returns topological_nodes, topological_parents

        len(topological_nodes) == num_node is a list the ids of the nodes in topological order

        topological_parents = [pa_1, ..., pa_num_nodes] each pa_i is a list that contains the ids
        of the parents according to the ordering in topological_nodes
        '''

        G = self._get_G()
        topological_nodes = list(range(self.num_nodes))

        topological_parents = []

        for i in range(self.num_nodes):
            topological_parents.append(list(G.predecessors(i)))

        return topological_nodes, topological_parents

    def sample(self, n_samples=1):

        nodes_list, parents_list = self.get_topological_nodes_pa()

        x = {}
        u = {}

        for obs_i, pa_i in zip(nodes_list, parents_list):
            if len(pa_i) == 0:
                xi, ui = self.sample_obs(obs_id=obs_i, n_samples=n_samples)
            else:
                parents_dict = {self.nodes_list[pa_ij]: x[self.nodes_list[pa_ij]] for pa_ij in pa_i}
                xi, ui = self.sample_obs(obs_id=obs_i, parents_dict=parents_dict, n_samples=n_samples)

            x[self.nodes_list[obs_i]] = xi
            u[self.nodes_list[obs_i]] = ui
        x = np.concatenate([x[k] for k in self.nodes_list], axis=1)
        u = np.concatenate([u[k] for k in self.nodes_list], axis=1)
        return x, u

    def _sample_noise(self, node_name, n_samples):
        u = np.array(self.noises_distr[node_name].sample(n_samples))
        if len(u.shape) == 0:
            u = np.array([[u]]).astype(np.float32)
        elif len(u.shape) == 1:
            u = u.reshape(-1, 1).astype(np.float32)
        return u

    def sample_obs(self, obs_id, parents_dict=None, n_samples=1, u=None):
        '''
        Only possible if the true Structural Eauations are known
        f = self.structural_eq[f'x{obs_id}']
        if u is None:
            u = np.array(self.noises_distr[f'u{obs_id}'].sample(n_samples))

        if not isinstance(parents_dict, dict):
            return f(u), u
        else:
            return f(u, **parents_dict), u
        '''
        assert obs_id < len(self.nodes_list)
        node_name = self.nodes_list[obs_id]
        lik = flatten(self.likelihoods)[obs_id]
        f = self.structural_eq[node_name]
        u_is_none = u is None
        if u_is_none:
            u = self._sample_noise(node_name, n_samples)
        x = f(u, **parents_dict) if isinstance(parents_dict, dict) else f(u)
        x = x.astype(np.float32)
        if lik.name in [Cte.CATEGORICAL]:
            x_out = np.zeros(x.shape[0], lik.domain_size)
            for i in range(x.shape[0]):
                x_out[i, x[i, 0]] = 1.

            x = x_out.copy()
            print(x)
            assert False

        return x, u

    def _get_set_nodes(self, x_I: Dict[str, float]):
        """

        Args:
            x_I: Dictionary of intervened nodes and intervened values

        Returns:
            List of parent nodes
            List of intervened-on nodes
            List of children nodes

        """
        parent_nodes = []
        children_nodes = []
        intervened_nodes = [self.nodes_list.index(i) for i in x_I.keys()]

        G = self._get_G()
        for id_ in intervened_nodes:
            ancestors = nx.algorithms.dag.ancestors(G, id_)
            ancestors = list_substract(ancestors, intervened_nodes)
            parent_nodes.extend(ancestors)

            descendants = nx.algorithms.dag.descendants(G, id_)
            descendants = list_substract(descendants, intervened_nodes)
            children_nodes.extend(descendants)

        return list(set(parent_nodes)), list(set(intervened_nodes)), list(set(children_nodes))

    def sample_intervention(self,
                            x_I: Dict[str, float],
                            n_samples: int = 1,
                            return_set_nodes: bool = False):

        """
        This method samples samples of all the nodes after intervening on x_I.
        The implementation only works for unidimensional nodes
        Args:
            x_I:
            n_samples:
            return_set_nodes:

        Returns:
        if return_set_nodes:
            return x_intervention, set_nodes
        else:
            return x_intervention
        """
        parent_nodes, intervened_nodes, children_nodes = self._get_set_nodes(x_I)

        nodes_list_all, parents_list_all = self.get_topological_nodes_pa()

        if self.has_ground_truth:
            x = {}
            node_dims = self.get_node_columns_in_X()
            for obs_i, pa_i in zip(nodes_list_all, parents_list_all):
                if obs_i in intervened_nodes:
                    obs_id_dim = node_dims[obs_i]
                    assert len(obs_id_dim) == 1, 'Interventions only implemented in unidimensional nodes'
                    xi_I = x_I[self.nodes_list[obs_i]]

                    xi = np.array([[xi_I]] * n_samples)
                else:
                    if len(pa_i) == 0:
                        xi, _ = self.sample_obs(obs_id=obs_i, n_samples=n_samples)
                    else:
                        parents_dict = {self.nodes_list[pa_ij]: x[self.nodes_list[pa_ij]] for pa_ij in pa_i}
                        xi, _ = self.sample_obs(obs_id=obs_i, parents_dict=parents_dict, n_samples=n_samples)

                x[self.nodes_list[obs_i]] = xi

            x_intervention = np.concatenate([x[k] for k in self.nodes_list], axis=1)
        else:
            x_intervention = None

        if return_set_nodes:
            set_nodes = {'parents': parent_nodes,
                         'intervened': intervened_nodes,
                         'children': children_nodes}

            return x_intervention, set_nodes
        else:
            return x_intervention

    def get_counterfactual(self,
                           x_factual: np.ndarray,
                           u_factual: np.ndarray,
                           x_I: Dict[str, float],
                           is_noise: bool = False,
                           return_set_nodes: bool = False):
        """
        This method gets the counterfactual of x_factual when we intervine on x_I.
        The implementation only works for unidimensional nodes
        Args:
            x_factual: Matrix with the factual samples [num_samples, num_dimensions]
            u_factual:   Matrix with the noise valules generating x_factual [num_samples, num_nodes]
            x_I:
            is_noise:
            return_set_nodes:

        Returns:
        if return_set_nodes:
            return x_cf, set_nodes
        else:
            return x_cf
        """
        is_tensor = isinstance(u_factual, torch.Tensor)
        if is_tensor:
            u_factual = u_factual.clone().cpu().numpy()
            x_factual = x_factual.clone().cpu().numpy()

        n_samples = u_factual.shape[0]

        _, intervened_nodes, children_nodes = self._get_set_nodes(x_I)

        nodes_list_all, parents_list_all = self.get_topological_nodes_pa()
        if self.has_ground_truth:
            x = {}
            if x_factual.shape[1] == self.total_num_dim_x0:
                if isinstance(x_factual, torch.TensorType):
                    x_factual = self._get_x_from_x0(x_factual).clone()
                else:
                    x_factual = self._get_x_from_x0(x_factual).copy()

            node_dims = self.get_node_columns_in_X()

            for obs_i, pa_i in zip(nodes_list_all, parents_list_all):
                if obs_i in intervened_nodes:
                    obs_id_dim = node_dims[obs_i]
                    assert len(obs_id_dim) == 1, 'Interventions only implemented in unidimensional nodes'
                    xi_I = x_I[self.nodes_list[obs_i]]

                    xi = (x_factual[:, [obs_i]] + xi_I) if is_noise else np.array([[xi_I]] * n_samples)
                else:
                    if len(pa_i) == 0:
                        xi, _ = self.sample_obs(obs_id=obs_i, u=u_factual[:, [obs_i]])
                    else:

                        parents_dict = {self.nodes_list[pa_ij]: x[self.nodes_list[pa_ij]] for pa_ij in pa_i}
                        xi, _ = self.sample_obs(obs_id=obs_i, parents_dict=parents_dict, u=u_factual[:, [obs_i]])

                x[self.nodes_list[obs_i]] = xi

            x_cf = np.concatenate([x[k] for k in self.nodes_list], axis=1)
        else:
            x_cf = None

        if return_set_nodes:
            set_nodes = {'intervened': intervened_nodes,
                         'children': children_nodes}

            return x_cf, set_nodes
        else:
            return x_cf

    def _create_data(self):
        """
        This method sets the value for self.X and self.U
        Returns: None

        """
        raise NotImplementedError

    def get_node_dimensions(self):
        """

        Returns: list with num_nodes elements. Each element contains the number of dimensions of each node
            node_dims: List[int]
            len(node_dims) == num_nodes
        """
        node_dims = []
        for lik_node_i in self.likelihoods:
            node_dims.append(sum([lik_ij.domain_size for lik_ij in lik_node_i]))

        return node_dims

    def get_dim_to_scale(self):
        """
        Get column indexes of X that need scaling
        Returns:

        """
        dims = []
        for lik_node_i in self.likelihoods:
            for lik_node_ij in lik_node_i:
                dims.extend(lik_node_ij.has_fit(True))

        return list(np.where(dims))

    def get_dim_to_scale_x0(self):
        """
        Get column indexes of X0 that need scaling
        Returns:

        """
        dims = []
        for lik_node_i in self.likelihoods:
            dims_i = []
            for lik_node_ij in lik_node_i:
                dims_i.extend(lik_node_ij.has_fit(True))

            remaining = self.largest_node_dim - len(dims_i)
            dims.extend([*dims_i, *[False, ] * remaining])

        return list(np.where(dims)[0])

    def get_node_columns_in_X(self, node_name=None):
        """
        Get column indexes of X  for each node. It returns a lists of lists. I.e., the list i represent the column
        indexes of X that represent the node i.
        Returns:
            List[List[int]]
        """
        node_dims_list = self.get_node_dimensions()
        cumsum = [0, *np.cumsum(node_dims_list)]
        node_columns = []
        for size_i, cumsum_i in zip(node_dims_list, cumsum[:-1]):
            node_columns.append(list(range(cumsum_i, cumsum_i + size_i)))
        if node_name is None:
            return node_columns
        else:
            return node_columns[self.nodes_list.index(node_name)]

    def fill_up_with_zeros(self, X):
        """

        Args:
            X: matrix with samples from the SCM

        Returns:
            X0: extended samples matrix
            mask_X0: mask for the columns of X in X0
            dim_of_x_in_x0: List[List[int]]. element i contains a List[int] with the column indexes in X0
            that correspond to the column indexses of X
        """
        node_dim_list = self.get_node_dimensions()
        node_cols = self.get_node_columns_in_X()
        dim_of_x_in_x0 = []

        X0 = np.zeros([X.shape[0], self.num_nodes * self.largest_node_dim])
        mask_X0 = np.zeros([1, self.num_nodes * self.largest_node_dim])
        for i, node in enumerate(range(self.num_nodes)):
            X0[:, i * self.largest_node_dim:(i * self.largest_node_dim + node_dim_list[i])] = X[:, node_cols[i]]
            mask_X0[:, i * self.largest_node_dim:(i * self.largest_node_dim + node_dim_list[i])] = 1.
            dim_of_x_in_x0.append(list(range(i * self.largest_node_dim, i * self.largest_node_dim + node_dim_list[i])))
        # in topological order (A, C, R, S)
        return X0, torch.tensor(mask_X0).type(torch.bool), dim_of_x_in_x0

    def get_intervention_list(self):
        '''
        nodes_to_intervene refer to the id in nodes_list
        '''

        # std_list = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]

        list_inter = []

        for node_name in self.nodes_to_intervene:
            cols = self.get_node_columns_in_X(node_name)
            assert len(cols) == 1, 'Interventions implemented only for unidimensional variables'
            x = self.X[:, cols]
            std = x.std()
            mean = x.mean()

            for i in self.std_list:
                list_inter.append(
                    ({node_name: float(np.round(mean + i * std, decimals=2))}, f'{i}_sigma'))
        return list_inter

    def set_intervention(self, x_I: Dict[str, float],
                         is_noise=False):
        """
        Set an intervention given by x_I.
        Args:
            x_I: Dictionary of node names and values to be intervene-on.
            is_noise: x_intervened =  x_original + value  if true else   x_intervened = value

        Returns:

        """
        self.x_I = {}
        self.I_noise = is_noise

        node_id_list = []

        for var, value in x_I.items():
            self.x_I[var] = value
            node_id_list.append(self.var_to_idx[var])
        self.adj_object.set_intervention(node_id_list)

    def diagonal_SCM(self):
        """
        Remove all parent-children edges from the SCM
        Returns:

        """
        self.x_I = {}

        self.adj_object.set_diagonal()

    def clean_intervention(self):
        """
        Resets the intervention
        Returns:

        """
        self.x_I = None
        self.I_noise = False
        self.adj_object.clean_intervention()

    def set_transform(self, transform):
        self.transform = transform

    def prepare_adj(self, normalize_A=None, add_self_loop=True):
        assert normalize_A is None, 'Normalization on A is not implemented'
        self.normalize_A = normalize_A
        self.add_self_loop = add_self_loop

        if add_self_loop:
            SCM_adj = np.eye(self.num_nodes, self.num_nodes)
        else:
            SCM_adj = np.zeros([self.num_nodes, self.num_nodes])

        for node_i, children_i in self.adj_edges.items():
            row_idx = self.nodes_list.index(node_i)
           # print('\nnode_i', node_i)
            for child_j in children_i:
            #    print('\tchild_j', child_j)
                SCM_adj[row_idx, self.nodes_list.index(child_j)] = 1
        # Create Adjacency Object
        self.dag = SCM_adj
        self.adj_object = Adjacency(SCM_adj)

    def prepare_data(self, normalize_A=None, add_self_loop=True, new_data = False):
#        print(f'\nPreparing data...')
        self.prepare_adj(normalize_A, add_self_loop)
        self._create_data(new_data)  # This should create X, and U
        self.X0, self.mask_X0, self.dim_of_x_in_x0 = self.fill_up_with_zeros(self.X)  # [800 x 48]

        self.total_num_dim_x0 = self.X0.shape[1]

    def _get_x0_dim_of_node_name(self, node_name):
        """

        Args:
            node_name: String name of the node for which we want to extract the column indexes in X0

        Returns:
            List[int]

        """
        node_idx = self.nodes_list.index(node_name)
        return self.dim_of_x_in_x0[node_idx]

    def __getitem__(self, index):

        x = self.X0[index].copy().astype(np.float32)
        u = torch.tensor(self.U[index].copy()).reshape(1, -1)
        edge_index = self.adj_object.edge_index
        edge_attr = self.adj_object.edge_attr

        x_i, edge_index_i, edge_attr_i = None, None, None
        if self.x_I is not None:  # Habemum intervention!
            x_i = x.copy()
            if self.I_noise == False:  # x_intervention = intervention_value

                if len(self.x_I) == 0:
                    for node_name, value in self.x_I.items():
                        dims_int = self._get_x0_dim_of_node_name(node_name)
                        x_i[dims_int] = value

                    edge_index = self.adj_object.edge_index_i
                    edge_attr = self.adj_object.edge_attr_i
                else:
                    for node_name, value in self.x_I.items():
                        dims_int = self._get_x0_dim_of_node_name(node_name)
                        x_i[dims_int] = value

                    edge_index_i = self.adj_object.edge_index_i
                    edge_attr_i = self.adj_object.edge_attr_i
            else:  # x_intervention = x_original + intervention_value
                for node_name, value in self.x_I.items():
                    dims_int = self._get_x0_dim_of_node_name(node_name)
                    x_i[dims_int] = x_i[dims_int] + value

                edge_index_i = self.adj_object.edge_index_i
                edge_attr_i = self.adj_object.edge_attr_i

        if self.transform:
            x = self.transform(x).view(self.num_nodes, -1)
            if x_i is not None: x_i = self.transform(x_i).view(self.num_nodes, -1)
        else:
            x = ToTensor()(x).view(self.num_nodes, -1)
            if x_i is not None: x_i = ToTensor()(x_i).view(self.num_nodes, -1)

        data = Data(x=x,
                    u=u,
                    mask=self.mask_X0.view(self.num_nodes, -1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_ids=torch.eye(self.num_nodes),
                    x_i=x_i,
                    edge_index_i=edge_index_i,
                    edge_attr_i=edge_attr_i,
                    num_nodes=self.num_nodes)

        return data

    def __len__(self):
        return len(self.X)

    def print_summary_X(self, columns=None):
        if columns is None:
            columns = list(range(self.num_dimensions))

        for c in columns:
            print(f'\n Dimension {c} | {self.node_per_dimension_list[c]}')
            if isinstance(self.X, torch.Tensor):
                x = self.X[:, c].numpy()
            else:
                x = self.X[:, c]

            my_str = f"min: {x.min():.3f} max: {x.max():.3f} mean: {x.mean():.3f} std: {x.std():.3f}"

            if len(np.unique(x)) < 10:
                uni = ' '.join([f"{a:.2f}" for a in np.unique(x)])
                my_str += f" unique: {uni}"

            print(my_str)

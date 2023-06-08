from torch import nn
import torch.nn.functional as F
from dowhy.gcm.graph import InvertibleFunctionalCausalModel
from dowhy.gcm import InvertibleStructuralCausalModel, EmpiricalDistribution
import dowhy.gcm as cy
from sklearn.preprocessing import StandardScaler
from dowhy.gcm.util.general import shape_into_2d
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm




class CausalDiffusionModel(InvertibleFunctionalCausalModel):
    """Main class for CDIM
    """

    def __init__(self,
                 hidden_dim =  64, 
                 use_positional_encoding = False, 
                 t_dim = 8, 
                 lr = 1e-4,
                 weight_decay = 0.001,
                 batch_size = 64, 
                 num_epochs = 10,
                 use_gpu_if_available = True,
                 verbose = False,
                 w = 0,
                 lambda_loss = 0,
                 T = 100,
                 betas = None,
                clip = False) -> None:
        self.device = torch.device("cuda" if (use_gpu_if_available and torch.cuda.is_available()) else "cpu")
        
        self._noise_model = None 
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding
        self.t_dim = t_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.w = w 
        self.lambda_loss = lambda_loss
        self.T = T
        self.betas = betas
        if self.betas is None:
            self.betas = np.linspace(1e-4,0.1,T)
        if len(self.betas) == T:
            self.betas = np.insert(self.betas,0,np.nan)
        self.betas = torch.from_numpy(self.betas)
        
        self.x_dim = None
        self.p_dim = None
        self.clip = clip

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        if self.x_dim is None:
            raise Exception("Need to fit model before drawing noise samples")
        return np.random.normal(size = (num_samples, self.x_dim))
    
    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        return self.decode_from_noise(parent_samples, noise_samples)

    
    def estimate_noise(self, x_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        return self.encode_from_obs(x_samples,parent_samples)
    
    def encode_from_obs(self, x: np.ndarray, parents: np.ndarray) -> np.ndarray:
        x, parents = shape_into_2d(x, parents)
        latent = self._prediction_model.encode_x_to_u(x, parents).cpu().numpy()
        
        if self.clip:
            sq_norm = np.linalg.norm(latent, axis = 1) ** 2
            indices = np.array((sq_norm > self.x_dim * 5)).flatten()
            if indices.sum() > 0:
                # print(x[indices,:])
                # print(parents[indices,:])
                # print(latent[indices,:])
                latent[indices,:] = (latent / np.sqrt(sq_norm).reshape(-1,1))[indices,:] * 5
            
        return latent
    
    def decode_from_noise(self, parents: np.ndarray, noise: np.ndarray) -> np.ndarray:
        parents, noise = shape_into_2d(parents, noise)
        pred_obs = self._prediction_model.decode_u_to_x(noise, parents)
        pred_obs = shape_into_2d(pred_obs)
        return pred_obs
    
    def encode_decode(self, x: np.ndarray, parents: np.ndarray) -> np.ndarray:
        u = self.encode_from_obs(x, parents)
        x_hat = self.decode_from_noise(parents, u)
        return x_hat
                      
    def fit(self,
            X, Y):
        """Fit diffusion models
        X : parent nodes values
        Y : current node values
        """
        parents, x = shape_into_2d(X, Y)
        self.x_dim = x.shape[1]
        self.p_dim = parents.shape[1]
        self._prediction_model = ep_theta(x_dim = self.x_dim,
                                           p_dim = self.p_dim, 
                                           hidden_dim = self.hidden_dim,
                                           use_positional_encoding = self.use_positional_encoding,
                                           t_dim = self.t_dim,
                                           lr = self.lr,
                                           weight_decay = self.weight_decay,
                                           batch_size = self.batch_size,
                                           num_epochs = self.num_epochs,
                                           verbose = self.verbose, 
                                           device = self.device,
                                           w = self.w,
                                           lambda_loss = self.lambda_loss,
                                           T = self.T,
                                           betas = self.betas,          
                                           )
        
        self._prediction_model.fit(x = x, parents = parents)
        if self.verbose:
            self.get_encoding_rmse(x, parents)
        

    def get_encoding_rmse(self, x, parents):
        x, parents= shape_into_2d(x, parents)
        x_hat = self.encode_decode(x, parents)
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        if self.verbose:
            if x.shape[1] == 1:
                fig,ax = plt.subplots()
                plt.scatter(x.flatten(), x_hat.flatten())
                plt.xlabel("Original X")
                plt.ylabel("Reconstructed X")
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.set_xlim(lims)
                ax.set_ylim(lims);
        return rmse

    def __str__(self) -> str:
        return f'General FCM using diffusion models'

    def clone(self):
        return CausalDiffusionModel()
                                            
        
class ep_theta(nn.Module):
    def __init__(self, 
                 x_dim,  
                 p_dim, 
                 hidden_dim,
                 use_positional_encoding,
                 t_dim, 
                 lr,
                 weight_decay,
                 batch_size, 
                 num_epochs,
                 verbose,                 
                 device,
                 w,
                 lambda_loss,
                 T,
                 betas):
        super(ep_theta, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_positional_encoding = use_positional_encoding
        self.device = device
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.loss_fn = nn.MSELoss()
        self.w = w
        self.lambda_loss = lambda_loss
        self.T = T

        temp = 1-betas
        temp[0] = 1
        alphas = np.cumprod(temp)
        
        self.register_buffer('sqrt_one_minus_betas', torch.sqrt(1-betas))
        self.register_buffer('sqrt_one_minus_alphas', torch.sqrt(1-alphas))
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        if self.use_positional_encoding:
            self.diffusion_embedding = TimeEmbedding(
                 dim = t_dim, proj_dim = hidden_dim, max_steps = self.T
            )
            self.fc1 = nn.Linear(x_dim + p_dim + hidden_dim,  hidden_dim)
        else:
            self.fc1 = nn.Linear(x_dim + p_dim + 1,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fcLast = nn.Linear(hidden_dim * 2, x_dim)
        self = self.to(self.device)

        self.dropout_prob = 0.1
        self.binom = torch.distributions.binomial.Binomial(probs = 1 - self.dropout_prob)
        
    def forward(self,x,p,t):
        
        if self.w != 0 :
            if self.training:       
                samp = self.binom.sample((p.shape[0],1)).to(self.device)
                p = torch.mul(p,samp)
        
        if self.use_positional_encoding:
            t = t.flatten()
            diffusion_step = self.diffusion_embedding(t-1) #subtract 1 since we zero index
            xpt = torch.cat([x, p, diffusion_step], 1)
        
        else:
            time =  torch.div(t, self.T).to(torch.float)
            if p is None:
                xpt = torch.cat([x, time], 1)
            else:
                xpt = torch.cat([x, p, time], 1)
        x = F.silu(self.fc1(xpt))        
        x = F.silu(self.fc2(x))
        x = self.fcLast(x)

        return x
    
    def sampling_forward(self, x,p,t):
        if self.w != 0:
            return (1+self.w)*self.forward(x,p,t) - (self.w)*self.forward(x,torch.zeros_like(p),t)
        else:
            return self.forward(x,p,t)
        
    def fit(self, x, parents):
        total_params = sum(p.numel() for p in self.parameters())
        if self.verbose:
            print(f"Encoding: {self.use_positional_encoding} Total Params: {total_params}")
        
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.num_samples = x.shape[0]
        self.x_scaler = StandardScaler()
        self.parent_scaler = StandardScaler()

        self.train_losses = []
        self.test_losses = []
       
        self.x_scaler = self.x_scaler.fit(x)
        x_scaled = self.x_scaler.transform(x)
        self.parent_scaler = self.parent_scaler.fit(parents)
        parents_scaled = self.parent_scaler.transform(parents)
 
        combined = np.hstack((x_scaled,parents_scaled))
        combined = torch.from_numpy(combined.astype(np.float32))
        train_loader = DataLoader(combined, batch_size=self.batch_size,
                                                     shuffle=True)
        if self.verbose:
            pbar = tqdm(range(1, self.num_epochs + 1), position = 1, leave=False)

            for i in pbar:
                self.train_loop(train_loader)
                pbar.set_description(f"Training Loss {self.train_losses[-1]:>.3f}")
        else:
            for i in range(1, self.num_epochs + 1):
                self.train_loop(train_loader)
        
    def pert_x_t(self, ep, x_0, t):
        x_t = torch.mul(x_0, self.sqrt_alphas[t]) + torch.mul(ep, self.sqrt_one_minus_alphas[t])
        return x_t.to(torch.float)

    def generate_pert_x(self, x, parents):
        t = torch.randint(1, self.T + 1,size = (x.shape[0],1))
        ep = torch.randn_like(x,requires_grad=False)
        
        noisy_x_t = self.pert_x_t(ep, x, t).to(self.device)
        t = t.to(self.device)
        return noisy_x_t, ep, t
    
    @torch.no_grad()
    def decode_u_to_x(self, u, parents, normalize = True):
        self.eval()
        if normalize:
            h = self.parent_scaler.transform(parents)
        hat_x, h = convert_numpy_to_torch(self.device, u, h)

        for t in range(self.T,0,-1):
            timestep = (t) * torch.ones((hat_x.shape[0],1)).to(torch.long).to(self.device)
            ep_val = self.sampling_forward(hat_x, h, timestep)#self(x, h, timestep)
            hat_x = hat_x / self.sqrt_one_minus_betas[t] + \
                (self.sqrt_one_minus_alphas[t-1] - self.sqrt_one_minus_alphas[t]/self.sqrt_one_minus_betas[t]) * ep_val

        hat_x = hat_x.cpu().numpy()
        if normalize:
            hat_x = self.x_scaler.inverse_transform(hat_x)
        return hat_x   
    

    def encode_x_to_u(self, x, parents, normalize = True, convert_to_torch = True, use_grad = False):
        with torch.set_grad_enabled(use_grad):
            self.eval()
            z, h = x, parents
            if normalize:
                z = self.x_scaler.transform(z)
                h = self.parent_scaler.transform(h)
            if convert_to_torch:
                z, h = convert_numpy_to_torch(self.device, z, h)
            for t in range(0, self.T):
                timestep = (t) * torch.ones((z.shape[0],1)).to(torch.long).to(self.device)

                ep_val = self.sampling_forward(z, h, timestep)

                z = self.sqrt_one_minus_betas[t+1] * z +\
                    (self.sqrt_one_minus_alphas[t+1] - \
                     self.sqrt_one_minus_alphas[t] * self.sqrt_one_minus_betas[t+1]) * ep_val
            return z

    def train_loop(self,loader):
        total_loss = 0 
        
        acc_size = 0
        for batch, data in enumerate(loader):
            self.train()
            self.zero_grad()
            # Temporarily only works for 1d data
            data = data.to(self.device)
            x, parents = data[:,[0]], data[:,1:]
            noisy_x_t, ep, t = self.generate_pert_x(x, parents)
            pred_ep = self.forward(noisy_x_t, parents, t)
            loss = self.loss_fn(pred_ep, ep)
            if self.lambda_loss > 0:
                pass
                # Only incorporate HSIC loss every 10 batches or on the last batch
                # if (batch-1) % 10 == 0 or batch == len(loader)-1:

                #     encoding = self.encode_x_to_u(x, 
                #                                   parents, 
                #                                   normalize = False, 
                #                                   convert_to_torch = False, 
                #                                   use_grad = True)
                   
                #     indep_reg =  HSIC(encoding, parents) 
                
                  
                #     loss = (loss + self.lambda_loss* indep_reg) / (1+self.lambda_loss)

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()*len(data)
            acc_size += len(data)
        
        self.train_losses.append(total_loss/acc_size)
            

# https://github.com/zalandoresearch/pytorch-ts/blob/81be06bcc128729ad8901fcf1c722834f176ac34/pts/model/time_grad/epsilon_theta.py#L83

# The MIT License (MIT)

# Copyright (c) 2020 Zalando SE

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class TimeEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(dim * 2, proj_dim * 2),
            nn.Mish(),
            nn.Linear(proj_dim * 2, proj_dim)
        )

    def forward(self, diffusion_step):
        
        x = self.embedding[diffusion_step]
        x = self.time_mlp(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

def convert_numpy_to_torch(device, *args):
    def converting(X: np.ndarray):
        torch_tensor = torch.from_numpy(X.astype(np.float32))
        if device is not None:
            return torch_tensor.to(device)
        else:
            return torch_tensor
    
    result = [converting(x) for x in args]

    if len(result) == 1:
        return result[0]
    else:
        return result
    
# def HSIC(data_x, data_y):
#     try: 
#         from KernelIndependence.KCI_torch import KCI_UInd
#     except:
#         raise ImportError("Missing KCI. TODO")
#     kci = KCI_UInd()
#     Kx, Ky = kci.kernel_matrix(data_x, data_y)
#     test_stat,_, _ = kci.HSIC_V_statistic(Kx, Ky)
#     return test_stat

def create_model_from_graph(g, params):
    diff_model = InvertibleStructuralCausalModel(g.copy())
    for node in g.nodes:
        if g.in_degree[node] == 0:
            diff_model.set_causal_mechanism(node, EmpiricalDistribution())
        else:
            diff_model.set_causal_mechanism(node, CausalDiffusionModel(**params))
    return diff_model


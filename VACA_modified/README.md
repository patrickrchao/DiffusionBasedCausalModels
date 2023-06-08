
#  VACA

Code repository for the paper "VACA: Designing Variational Graph Autoencoders for Interventional and Counterfactual Queries ([arXiv](https://arxiv.org/abs/2110.14690))". 
The implementation is based on [Pytorch](https://pytorch.org/), 
 [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and 
 [Pytorch Lightning](https://www.pytorchlightning.ai/). The repository contains the necessary resources to run the 
experiments of the paper. Follow the instructions below to download the German dataset.

## Installation

#### Option 1: Import the conda environment
```
conda env create -f environment.yml
```
#### Option 2: Commands

Create conda environment and activate it:

```
conda create --name vaca python=3.9 --no-default-packages
conda activate vaca 
```
Then install:


```
conda install pip
pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install pytorch-lightning==1.4.9
 pip install torchmetrics==0.6.2
pip install setuptools==59.5.0
pip install networkx==2.8.2
pip install matplotlib==3.5.2
pip install seaborn==0.11.2
```

**Note**: The German dataset is not contained in this repository. The first time you try to train on the German dataset, 
you will get an error with instructions on how to download and store it. Please follow the instructions, 
such that the code runs smoothly.

## Datasets
This repository contains 7 different SCMs:
    - ColliderSCM
    - MGraphSCM
    - ChainSCM
    - TriangleSCM
    - LoanSCM
    - AdultSCM
    - GermanSCM

Additionally, we provide the implementation of the first five SCMs with three different types of structural equations: 
linear (LIN), non-linear (NLIN) and non-additive (NADD). You can find the implementation of all the datasets inside the folder
`datasets`. To create all datasets at once run `python _create_data_toy.py` (this is optional since the datasets will be created as needed on the fly). 


#### How to create your custom Toy Datasets
We also provide a function to create custom ToySCM datasets. Here is an example of an SCM with 2 nodes

```
from datasets.toy import create_toy_dataset
from utils.distributions import *
dataset = create_toy_dataset(root_dir='./my_custom_datasets',
                             name='2graph',
                             eq_type='linear',
                             nodes_to_intervene=['x1'],
                             structural_eq={'x1': lambda u1: u1,
                                            'x2': lambda u2, x1: u2 + x1},
                             noises_distr={'x1': Normal(0,1),
                                           'x2': Normal(0,1)},
                             adj_edges={'x1': ['x2'],
                                        'x2': []},
                             split='train',
                             num_samples=5000,
                             likelihood_names='d_d',
                             lambda_=0.05)
```


For a complete example of training with a custom dataset, see the demo we included in the jupyter notebook `demo.ipynb`.

## Training

To train a model you need to execute the script `main.py`. For that, you need to specify three configuration files:
    - `dataset_file`: Specifies the dataset and the parameters of the dataset. You can overwrite the dataset parameters `-d`.
    - `model_file`: Specifies the model and the parameters of the model as well as the  optimizer.  You can overwrite  the model parameters with `-m` and the optimizer parameters with `-o`.
    - `trainer_file`:  Specifies the training parameters of the Trainer object from PyTorch Lightning.


For plotting results use `--plots 1`. For more information, run `python main.py --help`.



#### Examples

To train our VACA algorithm  on each of the synthetic graphs with linear structural equations (default value in `dataset_<NAME>`):


```
python main.py --dataset_file _params/dataset_adult.yaml --model_file _params/model_vaca.yaml
python main.py --dataset_file _params/dataset_loan.yaml --model_file _params/model_vaca.yaml
python main.py --dataset_file _params/dataset_chain.yaml --model_file _params/model_vaca.yaml
python main.py --dataset_file _params/dataset_collider.yaml --model_file _params/model_vaca.yaml
python main.py --dataset_file _params/dataset_mgraph.yaml --model_file _params/model_vaca.yaml
python main.py --dataset_file _params/dataset_triangle.yaml --model_file _params/model_vaca.yaml
```


You can also select a different SEM with the `-d` option and 
 - for linear (LIN) equations `-d equations_type=linear`,
 - for non-linear (NLIN) equations `-d equations_type=non-linear`, 
 - for non-additive (NADD) equation `-d equations_type=non-additive`. 
 
For example, to train the triangle graph with non linear SEM:
```
python main.py --dataset_file _params/dataset_triangle.yaml --model_file _params/model_vaca.yaml -d equations_type=non-linear
```


We can train our VACA algorithm on the German dataset:
```
python main.py --dataset_file _params/dataset_german.yaml --model_file _params/model_vaca.yaml
```


To run the CAREFL model:

```
python main.py --dataset_file _params/dataset_adult.yaml --model_file _params/model_carefl.yaml
python main.py --dataset_file _params/dataset_loan.yaml --model_file _params/model_carefl.yaml
python main.py --dataset_file _params/dataset_chain.yaml --model_file _params/model_carefl.yaml
python main.py --dataset_file _params/dataset_collider.yaml --model_file _params/model_carefl.yaml
python main.py --dataset_file _params/dataset_mgraph.yaml --model_file _params/model_carefl.yaml
python main.py --dataset_file _params/dataset_triangle.yaml --model_file _params/model_carefl.yaml
```
To run the MultiCVAE model:

```
python main.py --dataset_file _params/dataset_adult.yaml --model_file _params/model_mcvae.yaml
python main.py --dataset_file _params/dataset_loan.yaml --model_file _params/model_mcvae.yaml
python main.py --dataset_file _params/dataset_chain.yaml --model_file _params/model_mcvae.yaml
python main.py --dataset_file _params/dataset_collider.yaml --model_file _params/model_mcvae.yaml
python main.py --dataset_file _params/dataset_mgraph.yaml --model_file _params/model_mcvae.yaml
python main.py --dataset_file _params/dataset_triangle.yaml --model_file _params/model_mcvae.yaml
```



## How to load a trained model?
To load a trained model:
 - set the training flag to `-i 0`.
 - select configuration file of our training model, i.e. `hparams_full.yaml`
```
python main.py --yaml_file=PATH/hparams_full.yaml -i 0
```


## Load a model and train/evaluate counterfactual fairness
Load your model and add the flag `--eval_fair`. For example:

```
python main.py --yaml_file=PATH/hparams_full.yaml -i 0 --eval_fair --show_results
```



## TensorBoard visualization

You can track different metrics during (and after) training using TensorBoard. 
For example, if the root folder of the experiments is `exper_test`, we can run the following
command in a terminal

```
tensorboard --logdir exper_test/   
```
to display the logs of all experiments contained in such folder. Then, we go to our favourite browser 
and go to `http://localhost:6006/` to visualize all the results. 

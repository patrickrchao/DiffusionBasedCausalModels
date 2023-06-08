

<!-- TITLE -->
#  &nbsp; **Interventional and Counterfactual Inference with Diffusion Models** &nbsp; 

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2302.00860-b31b1b)](https://arxiv.org/abs/2302.00860)
</div>


<!-- DESCRIPTION -->
## Abstract

We consider the problem of answering observational, interventional, and counterfactual queries in a causally sufficient setting where only observational data and the causal graph are available. Utilizing the recent developments in diffusion models, we introduce diffusion-based causal models (DCM) to learn causal mechanisms, that generate unique latent encodings. These encodings enable us to directly sample under interventions and perform abduction for counterfactuals. Diffusion models are a natural fit here, since they can encode each node to a latent representation that acts as a proxy for exogenous noise. Our empirical evaluations demonstrate significant improvements over existing state-of-the-art methods for answering causal queries. Furthermore, we provide theoretical results that offer a methodology for analyzing counterfactual estimation in general encoder-decoder models, which could be useful in settings beyond our proposed approach. 



## Code

### Installation
Create a conda environment with the command:
```bash
conda env create -f environment.yml
```

<!-- MINIMUM WORKING EXAMPLE -->
### Example with Custom Data
Diffusion based Causal Models (DCMs) can answer causal queries using observational data and the causal graph. We may consider an example with a triangle graph, where X1 causes X2, and both X1 and X2 cause X3. We may first generate a dataset.
```python
import numpy as np 
import pandas as pd
import networkx as nx 
from model.diffusion import create_model_from_graph
import dowhy.gcm as cy
from dowhy.gcm import draw_samples, interventional_samples, counterfactual_samples

n = 1000
# Make dataset
x1 = np.random.normal(size=(n))
x2 = x1 + np.random.normal(size=(n)) 
x3 = x1 + x2 + np.random.normal(size=(n)) 
factual = pd.DataFrame({"x1" : x1, "x2" : x2, "x3" : x3})

# Make Graph
graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2','x3')])
```

Next, we specify parameters for our DCMs, create the model, and fit the model on the data.
```python
params = {'num_epochs' : 200,
          'lr' : 1e-4,
          'batch_size': 64,
          'hidden_dim' : 64}

diff_model = create_model_from_graph(graph, params)

cy.fit(diff_model, factual)
```

After we fit our model, we can ask causal queries. For example, we may perform *observational queries*:
```python
# Observational Query
obs_samples = draw_samples(diff_model, num_samples = 20)
```

We may also perform *interventional queries*:
```python
# Interventional Query
intervention = {"x1": lambda x: 2, "x2": lambda x: x - 1}
int_samples = interventional_samples(diff_model, intervention, num_samples_to_draw=20)
```

And we may perform *counterfactual queries*:
```python
# Counterfactual Query
cf_estimates = counterfactual_samples(diff_model, intervention, observed_data = factual)
cf_estimates.head()
```

For more examples, see `mvp.ipynb`. To rerun our experiments in the paper, run the following command:
```bash
python3 all_exp.py
```

<!-- CITATION -->
## Citation

If you find this work useful, please cite:

```bibtex
@misc{chao2023interventional,
      title={Interventional and Counterfactual Inference with Diffusion Models}, 
      author={Patrick Chao and Patrick Blöbaum and Shiva Prasad Kasiviswanathan},
      year={2023},
      eprint={2302.00860},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
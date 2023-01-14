# FedPop: A Bayesian Approach for Personalised Federated Learning

This repository accompanies the paper. Here you can find all the code, needed to launch experiments.
Below you can see an instruction, on how one should proceed to reproduce the results.

### Preparation

1. Create a virtual environment. For example: ```python3 -m venv venv```
2. Activate it: ```source venv/bin/activate```
3. Install requirements: ```pip install -r requitements.txt```
4. Make sure, that tests work: ```pytest```

### Launch Image-datasets experiments

!!!! NOTE, that we have not attached model weights since they are heavy. Make sure you run appropriate experiment beforehand
to plot pictures from notebooks! !!!!

#### Standard setup

Then, to launch main experiments, one should specify the config file (which is stored in ./configs) and run main.py script.
Example:
```python main.py --config ./configs/cifar_10_2.yml```

Depending on the used dataset, the resulting model, config file and metrics will be saved to either experiment_logs (
cifar10) or experiment_logs100 (cifar100) folder.
Folders will be enumerated, depending on the number of the experiment.

#### Small data setup

If you want to launch experiments with heterogeneous data, use the following
command: ```python run_exps_heterogeneous_data.py```

For FedRep, DITTO, APFL, and others we used the code from the
official [FedRep](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf) repository.
In some places of our code (notebooks), there are links to the FedRep. Make sure you cloned this repo next to our
project folder.

### Toy example

The toy example (inspired
from [Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf))
can be found in notebooks/Toy_example.ipynb.
There, you can play with it and receive plots.

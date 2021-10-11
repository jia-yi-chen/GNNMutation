# GNNMutation

A software testing tool intended for injecting faults into trained GNN models. Currently, the tool has been tested on two popular GNN models using Cora Dataset:
1) [GCN](https://github.com/tkipf/pygcn)
2) [GAT](https://github.com/Diego999/pyGAT)


## Requirements 
* python3
* pytorch
* numpy
* scipy


## Getting Started

Mutation operators are located in:  $GNN_MODEL_ROOT$/mutators/graph_mutator_operators.py 

### Testing GCN 
Step 1: set the root as ./pygcn-master.
Step 2: examples
```
python3 pygcn-master/mutators/mutation_testing.py --mutation_ratio 0.2 --repetition 5 --mutation_operator 'All'
```


### Testing GAT
Step 1: set the root as ./pyGAT-master.
Step 2: examples
```
python3 pyGAT-master/mutators/mutation_testing.py --mutation_ratio 0.2 --repetition 5 --mutation_operator 'All'
```





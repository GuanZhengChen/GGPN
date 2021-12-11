# GGPN
Source code for "Multi-Relational Graph Representation Learning with Bayesian Gaussian Process Network"

## Prepare the datasets
```
./preprocess.sh
```

## Run link prediction
```
python -m torch.distributed.launch --nproc_per_node=number_of_gpu run_link_prediction.py -data dataset_name
```
## Run entity classification
```
python -m torch.distributed.launch --nproc_per_node=number_of_gpu run_node_classification.py -data dataset_name
```
#prepare
./preprocess.sh

# run link prediction
python -m torch.distributed.launch --nproc_per_node=number_of_gpu run_link_prediction.py -data dataset_name

# run entity classification
python -m torch.distributed.launch --nproc_per_node=number_of_gpu run_node_classification.py -data dataset_name
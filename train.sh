# Semanticist training
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_xl.yaml
# or use torchrun
torchrun --nproc-per-node=8 train_net.py --cfg configs/tokenizer_xl.yaml
# or use submitit
python submitit_train.py --ngpus=8 --nodes=1 --partition=xxx --config configs/tokenizer_xl.yaml

# ϵLlamaGen training
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/autoregressive_xl.yaml
# or use torchrun
torchrun --nproc-per-node=8 train_net.py --cfg configs/autoregressive_xl.yaml
# or use submitit
python submitit_train.py --ngpus=8 --nodes=1 --partition=xxx --config configs/autoregressive_xl.yaml
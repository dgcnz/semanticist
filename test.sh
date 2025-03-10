# Semanticist testing
accelerate launch --config_file=configs/onenode_config.yaml test_net.py --model ./output/tokenizer/models_xl --step 250000 --cfg_value 3.0 --test_num_slots 256
# or use torchrun
torchrun --nproc-per-node=8 test_net.py --model ./output/tokenizer/models_xl --step 250000 --cfg_value 3.0 --test_num_slots 256
# or use submitit
python submitit_test.py --ngpus=8 --nodes=1 --partition=xxx --model ./output/tokenizer/models_xl --step 250000 --cfg_value 3.0 --test_num_slots 256

# ϵLlamaGen testing
accelerate launch --config_file=configs/onenode_config.yaml test_net.py --model ./output/autoregressive/models_xl --step 250000 --cfg_value 5.0 --ae_cfg 1.0 --test_num_slots 32
# or use torchrun
torchrun --nproc-per-node=8 test_net.py --model ./output/autoregressive/models_xl --step 250000 --step 250000 --cfg_value 5.0 --ae_cfg 1.0 --test_num_slots 32
# or use submitit
python submitit_test.py --ngpus=8 --nodes=1 --partition=xxx  --step 250000 --cfg_value 5.0 --ae_cfg 1.0 --test_num_slots 32
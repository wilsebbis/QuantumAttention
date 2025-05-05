#/bin/sh

root_dir="."
cd ${root_dir}

model_file_path="results/LMTAD_Quantum/porto/outlier_False/n_layer_1_n_head_1_n_embd_32_lr_0.0001_integer_poe_False/ckptepoch_9_batch_58.pt"

python -m trajectory_code.eval_porto_quantum --model_file_path ${model_file_path} --small_data_subset
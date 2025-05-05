#!/bin/sh

dataset="porto"
batch_size=4

#training configuration
eval_interval=20
log_interval=1
max_iters=10

#model config
block_size=16
n_layer=1
n_head=1
n_embd=32
dropout=0.1
lr_decay_iters=50
beta1=0.9
beta2=0.99
min_lr=1e-5
lr=1e-4
warmup_iters=5
weight_decay=0.01
decay_lr=True
grad_clip=1.0

if [[ "${dataset}" == "porto" ]] ; then
    data_dir=".data"
    data_file_name="porto_processed"
    out_dir="./results/LMTAD_Quantum/porto"
    outlier_days=0
    output_file_name=None
    features="no_features"
    grid_leng=10
fi

export PYTHONPATH=$(pwd):$PYTHONPATH

python -m trajectory_code.train_LMTAD_quantum \
    --data_dir ${data_dir} --data_file_name ${data_file_name} --dataset ${dataset} --outlier_days ${outlier_days} \
    --features ${features} --batch_size ${batch_size} --grid_leng ${grid_leng} \
    --out_dir ${out_dir} --output_file_name ${output_file_name} \
    --eval_interval ${eval_interval} --log_interval ${log_interval} --max_iters ${max_iters} \
    --block_size ${block_size} --n_layer ${n_layer} --n_head ${n_head} --n_embd ${n_embd} --dropout ${dropout} \
    --lr_decay_iters ${lr_decay_iters} --beta1 ${beta1} --beta2 ${beta2} --min_lr ${min_lr} --lr ${lr} \
    --warmup_iters ${warmup_iters} --weight_decay ${weight_decay} --decay_lr ${decay_lr} --grad_clip ${grad_clip} \
    --debug --small_data_subset
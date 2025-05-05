import time
import os
import math
import argparse
from contextlib import nullcontext
from typing import List
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from .datasets import (VocabDictionary, 
                      PortoConfig,
                      PortoDataset)

from .models import (
    LMTAD_Quantum_Config,
    LMTAD_Quantum,
)

from .utils import (log, seed_all, 
                   save_file_name_pattern_of_life, 
                   save_file_new_datset, 
                   save_file_name_porto,
                   save_file_name_trial0)

from .eval_porto import eval_porto
from .metrics import get_metrics, get_per_user_metrics

def get_parser():
    """argparse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_file_name', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--grid_leng', type=int, default=25)
    parser.add_argument('--dataset', type=str, default="porto", choices=["porto"])
    parser.add_argument('--include_outliers', action='store_true', required=False)
    parser.add_argument('--outlier_days', type=int, default=14)
    # parser.add_argument('--skip_gps', type=bool, default=True)
    parser.add_argument('--features', type=str, default="place")
    parser.add_argument('--small_data_subset', action='store_true', help='Use a small subset of the data for fast debugging')

    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--output_file_name', type=str, default="")

    parser.add_argument('--eval_interval', type=int, default=250)
    # parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--log_file', type=str, default="")

    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    # use simple poe (1, 2, 3), instead of the sin and cosine
    parser.add_argument('--integer_poe', action='store_true', required=False)

    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--lr_decay_iters', type=int, default=600000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--min_lr', type=float, default=6e-6)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_iters', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--decay_lr', type=bool, default=True)

    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_false')
    parser.add_argument('--debug', action='store_true', required=False)

    args = parser.parse_args()
    return args


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(args, model, test_dataloader, device, ctx):
    out = {}
    model.eval()
    losses = torch.zeros(len(test_dataloader))
    for batch, data in enumerate(test_dataloader):
        X = data["data"][:, :-1].contiguous().to(device)
        Y = data["data"][:, 1:].contiguous().to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses[batch] = loss.item()
    model.train()
    return losses.mean()

@torch.no_grad()
def model_eval(args,
         epoch,
         iter_num, 
         model, 
         optimizer, 
         model_conf, 
         dataset_config, 
         test_dataloader, 
         device, 
         ctx, 
         best_val_loss, 
         metric_results,
         **kwargs):
    """run evaluation on the eval set"""
    model.eval()
    val_loss = estimate_loss(args, model, test_dataloader, device, ctx)
    log(f"|step {iter_num}:  val loss {val_loss:.4f}|", args.log_file)
    saved_model_eval = False
        
    best_val_loss = val_loss
    if iter_num > 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_config': model_conf,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'dataset_config': dataset_config,
            "args": args
        }

        log_output = f"\nsaving checkpoint to {args.out_dir}"
        if args.output_file_name != "":
            ckpt_name = f"ckpt_{args.output_file_name}"
        else:
            ckpt_name = f"ckpt"

        ckpt_name += f"epoch_{epoch}_batch_{iter_num}.pt"

        torch.save(checkpoint, os.path.join(args.out_dir, ckpt_name))

        # if args.debug and args.dataset == "porto":   
        #     results = eval_porto(model=model, device=device, dataloader=kwargs["dataloader"])
        #     df_results = pd.DataFrame(results)            
            
        #     (_, _, _, _, f1, pr_auc), treshold = get_metrics(df_results[df_results["outlier"] != "detour outlier"], "log_perplexity")

        #     metric_results["model_number"].append(iter_num)
        #     metric_results["f1_rs"].append(f1)
        #     metric_results["pr_rs"].append(pr_auc)

        #     log_output += f"\n| route switching outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f}"

        #     (_, _, _, _, f1, pr_auc), treshold = get_metrics(df_results[df_results["outlier"] != "route switch outlier"], "log_perplexity")
        #     metric_results["f1_detour"].append(f1)
        #     metric_results["pr_auc_detour"].append(pr_auc)
        #     log_output += f"| detour outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f} |\n"
        
        log(log_output, args.log_file)
    
    saved_model_eval = True
    model.train()
    return best_val_loss, saved_model_eval

def main(args):
    """train orchastration"""

    log("Starting training orchestration...", args.log_file)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' or 'mps' else torch.amp.autocast(device_type=device, dtype=ptdtype)


    # if args.dataset == "porto":
    args.include_outliers = False

    dataset_config = PortoConfig()
    dataset_config.file_name = args.data_file_name
    dataset_config.outlier_level = 5
    dataset_config.outlier_prob = 0.1
    dataset_config.outlier_ratio = 0.05
    dataset_config.outliers_list = ["route_switch", "detour"]
    dataset_config.include_outliers = args.include_outliers

    args.out_dir = f"{args.out_dir}/outlier_{dataset_config.include_outliers}/n_layer_{args.n_layer}_n_head_{args.n_head}_n_embd_{args.n_embd}_lr_{args.lr}_integer_poe_{args.integer_poe}"

    os.makedirs(f"{args.out_dir}", exist_ok=True)

    output_file_name = save_file_name_porto(dataset_config)
    log_file = f"{args.out_dir}/log_{output_file_name}.txt"
    args.log_file = log_file
    args.output_file_name = output_file_name

    with open(args.log_file, "w") as f:
        f.write("")

    # log("Loading dataset...", args.log_file)
    dataset = PortoDataset(dataset_config)
    log(f"output file name: {args.output_file_name}", args.log_file)

    args.block_size = dataset_config.block_size
    args.features = []

    train_indices, val_indices = dataset.partition_dataset()

    # if args.small_data_subset:
    #     print("Using a small subset of the dataset for debugging...")
    #     train_indices = train_indices[: len(train_indices) // 1000 ]
    #     val_indices = val_indices[: len(val_indices) // 1000 ]

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(train_indices))
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(val_indices))

    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size, log_file=args.log_file,
                  bias=False, vocab_size=len(dataset.dictionary), dropout=args.dropout, pad_token=dataset.dictionary.pad_token(), logging=True, integer_poe=args.integer_poe)

    model_conf = LMTAD_Quantum_Config(**model_args)
    model = LMTAD_Quantum(model_conf)

    model = model.to(device)
    model.train()

    if not args.compile:
        log("Compiling the model... (this may take a while)", args.log_file)
        print("Compiling the model... (this may take a while)")
        log(f"compiling the model... (takes a ~minute)", args.log_file)
        model = torch.compile(model)

    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    best_val_loss = 1e9
    t0 = time.time()
    train_losses = []
    valid_losses = []
    cumul_train_loses = 0
    cumulation = 1
    save_model_count = 3
    metric_results = defaultdict(list)

    eval_outliers_kwargs = None

    if args.debug and args.dataset == "porto":
        eval_outliers_kwargs = {}
        test_dataset_config = PortoConfig(**vars(dataset_config))
        test_dataset_config.include_outliers = True
        test_dataset_config.outlier_level = 3
        test_dataset_config.outlier_prob = 0.1
        test_dataset_config.outlier_ratio = 0.05
        test_dataset_config.outliers_list = ["route_switch", "detour"]

        log('loading the metrics test dataset', args.log_file)
        test_dataset = PortoDataset(test_dataset_config)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=dataset.collate)

        eval_outliers_kwargs["dataloader"] = test_dataloader
        eval_outliers_kwargs["test_dataset_config"] = test_dataset_config

    iter_num = 0
    for epoch in range(args.max_iters):
        save_model = False
        log('-' * 85, args.log_file)
        for batch_id, data in enumerate(train_dataloader):

            lr = get_lr(iter_num, args) if args.decay_lr else args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            inputs = data["data"][:, :-1].contiguous().to(device)
            targets = data["data"][:, 1:].contiguous().to(device)

            if batch_id % (len(train_dataloader) - 1) == 0:
                best_val_loss, saved_model_eval = model_eval(args,
                                                       epoch, 
                                                       batch_id, 
                                                       model, 
                                                       optimizer, 
                                                       model_conf, 
                                                       dataset_config, 
                                                       val_dataloader, 
                                                       device, 
                                                       ctx, 
                                                       best_val_loss,
                                                       metric_results,
                                                       **eval_outliers_kwargs)

                if saved_model_eval:
                    save_model = saved_model_eval

                train_losses.append(cumul_train_loses/cumulation)
                valid_losses.append(best_val_loss.item())
                cumulation = 1
                cumul_train_loses = 0

            with ctx:
                logits, loss = model(inputs, targets)

            scaler.scale(loss).backward()
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            cumul_train_loses += loss.item()
            if batch_id % args.log_interval == 0:
                log(f"|epoch {epoch+1}/{args.max_iters} | batch {batch_id+1}/{len(train_dataloader)}: loss {loss.item():.4f} \t| time {dt*1000:.2f}ms|", args.log_file)

            cumulation += 1
            iter_num +=1 

        if not save_model:
            save_model_count +=1
        else:
            save_model_count = 0

        if save_model_count >= 5+1:
            log(f'stopped training because the loss did not improve {save_model_count-1} times', args.log_file)

        log('-' * 85, args.log_file)
        log(f"|save_model_count: {save_model_count} | lr: {lr}",  args.log_file)

    losses_dict = {"train": train_losses, "val":valid_losses}
    losses_dict = pd.DataFrame(losses_dict)
    losses_dict.to_csv(f"{args.out_dir}/losses_{args.output_file_name}.tsv", sep="\t")

    metric_results_df = pd.DataFrame(metric_results)          
    metric_results_df.to_csv(
            f"{args.out_dir}/metrics_results.tsv", 
            index=False,
            sep="\t"
        )

if __name__ == "__main__":
    args = get_parser()

    main(args)
"""
References: Inspired by LM-TAD (https://github.com/jonathankabala/LMTAD), which is inspired by the nanoGTPT repository found at this location: https://github.com/karpathy/nanoGPT/tree/master
"""

import math
import inspect
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..utils import log

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class QuantumKernelCircuit:
    def __init__(self):
        self.a = torch.tensor(0.0)
        self.b = torch.tensor(0.0)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.a.item(), 0)
        self.circuit.ry(self.b.item(), 1)
        self.circuit.cx(0, 1)

    def compute_similarity(self, a, b):
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        qc.ry(b, 1)
        qc.cx(0, 1)
        sv = Statevector(qc)
        zz_op = SparsePauliOp.from_list([("ZZ", 1)])
        return np.real(sv.expectation_value(zz_op))
        

class QuantumAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.kernel = QuantumKernelCircuit()
        self.seq_len = config.block_size

    def forward(self, x):
        print("Computing quantum attention...")
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        B, T, D = Q.shape
        attention_scores = torch.zeros((B, T, T), device=x.device)
        for b in range(B):
            Qb = Q[b].mean(dim=1).to(torch.float32).detach().cpu().numpy()
            Kb = K[b].mean(dim=1).to(torch.float32).detach().cpu().numpy()
            for i in range(T):
                for j in range(T):
                    score = self.kernel.compute_similarity(Qb[i], Kb[j])
                    attention_scores[b, i, j] = score
        weights = torch.softmax(attention_scores, dim=-1)
        print("Quantum attention weights computed.")
        return torch.bmm(weights, V)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class QETBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = QuantumAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
class PositionalEncoding(nn.Module):
    """positional embedding layer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

@dataclass
class LMTAD_Quantum_Config:
    """config dataclasss for the model"""
    block_size: int = 1024
    vocab_size: int = 1000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pad_token: int = 8217
    log_file: str = ""
    logging: bool = False
    integer_poe: bool = False

class LMTAD_Quantum(nn.Module):    
    def __init__(self, config: LMTAD_Quantum_Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config:LMTAD_Quantum_Config = config
        print("Initializing LMTAD_Quantum model...")

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if not self.config.integer_poe else PositionalEncoding(self.config.n_embd, self.config.dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([QETBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        print("Weight tying complete.")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        message=f"number of parameters: {self.get_num_params()/1e6:.3f}M"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        print("Model parameter initialization complete.")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.integer_poe:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):

        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """forward method"""

        print("Running forward pass...")
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        print("Embeddings and positional encodings computed.")

        # ipdb.set_trace()
        if not self.config.integer_poe:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        else:
            x = self.transformer.wpe(tok_emb)

        # ipdb.set_trace()

        for i, block in enumerate(self.transformer.h):
            print(f"Passing through QETBlock {i+1}/{len(self.transformer.h)}")
            x = block(x)       
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            # import pdb
            # pdb.set_trace()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x)
            loss = None
            print("Forward pass complete. Returning logits and loss.")

        return logits, loss
       
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        message=f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        message=f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        message=f"using fused AdamW: {use_fused}"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
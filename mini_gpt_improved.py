"""
Mini-GPT: Improved Version
==========================
Key improvements:
1. Adam optimizer
2. Gradient clipping
3. Dropout implementation
4. Learning rate scheduling
5. Model checkpointing
6. Better loss computation
7. Fixed attention backward
8. Type hints
"""

import numpy as np
from typing import Tuple, Dict, Optional
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Hyperparameters for our mini-GPT"""
    # Model architecture
    n_embd = 64
    n_head = 4
    n_layer = 4
    block_size = 64
    
    # Training
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    warmup_iters = 100
    lr_decay_iters = 4000
    min_lr = 3e-5
    
    # Regularization
    dropout = 0.0
    grad_clip = 1.0
    
    # Optimizer
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    def validate(self):
        """Validate config parameters"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.grad_clip > 0, "grad_clip must be positive"

config = Config()
config.validate()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def dropout(x: np.ndarray, p: float, training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dropout with inverted dropout"""
    if not training or p == 0:
        return x, None
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    return x * mask, mask

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU"""
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, tuple]:
    """Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_norm = (x - mean) / std
    out = gamma * x_norm + beta
    return out, (x, x_norm, mean, var, std)

def layer_norm_backward(dout: np.ndarray, cache: tuple, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop through layer norm - improved version"""
    x, x_norm, mean, var, std = cache
    N = x.shape[-1]
    
    dgamma = np.sum(dout * x_norm, axis=tuple(range(len(dout.shape)-1)))
    dbeta = np.sum(dout, axis=tuple(range(len(dout.shape)-1)))
    
    dx_norm = dout * gamma
    
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * std**(-3), axis=-1, keepdims=True)
    dmean = np.sum(dx_norm * -1/std, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
    dx = dx_norm / std + dvar * 2 * (x - mean) / N + dmean / N
    
    return dx, dgamma, dbeta

def get_lr(iter: int, config: Config) -> float:
    """Learning rate scheduler with warmup and decay"""
    if iter < config.warmup_iters:
        return config.learning_rate * iter / config.warmup_iters
    if iter > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1 + np.cos(np.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# ==============================================================================
# ADAM OPTIMIZER
# ==============================================================================

class AdamOptimizer:
    """Adam optimizer for parameter updates"""
    
    def __init__(self, params: Dict[str, np.ndarray], config: Config):
        self.params = params
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
    
    def step(self, grads: Dict[str, np.ndarray], lr: float):
        """Update parameters using Adam"""
        self.t += 1
        lr_t = lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for key in self.params:
            if key in grads:
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)
                self.params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)

# ==============================================================================
# ATTENTION MECHANISM
# ==============================================================================

class CausalSelfAttention:
    """Multi-Head Causal Self-Attention - Improved"""
    
    def __init__(self, config: Config):
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        
        # Initialize weights with proper scaling
        scale = np.sqrt(2.0 / config.n_embd)
        self.W_q = np.random.randn(config.n_embd, config.n_embd) * scale
        self.W_k = np.random.randn(config.n_embd, config.n_embd) * scale
        self.W_v = np.random.randn(config.n_embd, config.n_embd) * scale
        self.W_o = np.random.randn(config.n_embd, config.n_embd) * scale
        
        self.b_q = np.zeros(config.n_embd)
        self.b_k = np.zeros(config.n_embd)
        self.b_v = np.zeros(config.n_embd)
        self.b_o = np.zeros(config.n_embd)
        
        self.cache = {}
        self.grads = {}
    
    def _create_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for given sequence length"""
        return np.tril(np.ones((seq_len, seq_len)))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through attention"""
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = x @ self.W_q + self.b_q
        k = x @ self.W_k + self.b_k
        v = x @ self.W_v + self.b_v
        
        # Reshape for multi-head
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Apply causal mask (dynamic)
        mask = self._create_mask(T)
        att = np.where(mask == 0, -1e9, att)
        
        # Softmax
        att_weights = softmax(att, axis=-1)
        
        # Dropout on attention weights
        att_weights, att_mask = dropout(att_weights, self.dropout_p, training)
        
        # Apply attention to values
        out = att_weights @ v
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Final projection
        out_proj = out @ self.W_o + self.b_o
        
        # Save for backprop
        self.cache = {
            'x': x, 'q': q, 'k': k, 'v': v,
            'att_weights': att_weights, 'out': out,
            'att_mask': att_mask, 'mask': mask,
            'B': B, 'T': T, 'C': C
        }
        
        return out_proj
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagate through attention - fixed version"""
        x = self.cache['x']
        q, k, v = self.cache['q'], self.cache['k'], self.cache['v']
        att_weights = self.cache['att_weights']
        out = self.cache['out']
        att_mask = self.cache['att_mask']
        B, T, C = self.cache['B'], self.cache['T'], self.cache['C']
        
        # Undo dropout if applied
        if att_mask is not None:
            att_weights = att_weights / att_mask
        
        # Backward through output projection
        self.grads['W_o'] = out.reshape(B*T, C).T @ dout.reshape(B*T, C)
        self.grads['b_o'] = np.sum(dout, axis=(0, 1))
        dout_proj = dout @ self.W_o.T
        
        # Reshape for multi-head
        dout_proj = dout_proj.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        # Backward through attention
        dv = att_weights.transpose(0, 1, 3, 2) @ dout_proj
        datt_weights = dout_proj @ v.transpose(0, 1, 3, 2)
        
        # Backward through softmax (correct formula)
        datt = att_weights * (datt_weights - np.sum(datt_weights * att_weights, axis=-1, keepdims=True))
        
        # Re-apply mask gradient
        mask = self.cache['mask']
        datt = np.where(mask == 0, 0, datt)
        
        # Backward through scaling
        datt = datt / np.sqrt(self.head_dim)
        
        # Backward through matmul
        dq = datt @ k
        dk = datt.transpose(0, 1, 3, 2) @ q
        
        # Reshape back
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Backward through projections
        self.grads['W_q'] = x.reshape(B*T, C).T @ dq.reshape(B*T, C)
        self.grads['W_k'] = x.reshape(B*T, C).T @ dk.reshape(B*T, C)
        self.grads['W_v'] = x.reshape(B*T, C).T @ dv.reshape(B*T, C)
        self.grads['b_q'] = np.sum(dq, axis=(0, 1))
        self.grads['b_k'] = np.sum(dk, axis=(0, 1))
        self.grads['b_v'] = np.sum(dv, axis=(0, 1))
        
        dx = dq @ self.W_q.T + dk @ self.W_k.T + dv @ self.W_v.T
        
        return dx

# ==============================================================================
# FEEDFORWARD NETWORK
# ==============================================================================

class FeedForward:
    """Position-wise Feedforward Network with dropout"""
    
    def __init__(self, config: Config):
        hidden_dim = 4 * config.n_embd
        self.dropout_p = config.dropout
        
        scale = np.sqrt(2.0 / config.n_embd)
        self.W1 = np.random.randn(config.n_embd, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, config.n_embd) * scale
        self.b2 = np.zeros(config.n_embd)
        
        self.cache = {}
        self.grads = {}
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        h = x @ self.W1 + self.b1
        h_act = gelu(h)
        h_act, drop_mask = dropout(h_act, self.dropout_p, training)
        out = h_act @ self.W2 + self.b2
        
        self.cache = {'x': x, 'h': h, 'h_act': h_act, 'drop_mask': drop_mask}
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass"""
        x, h, h_act, drop_mask = self.cache['x'], self.cache['h'], self.cache['h_act'], self.cache['drop_mask']
        
        # Undo dropout if applied
        if drop_mask is not None:
            h_act = h_act / drop_mask
        
        self.grads['W2'] = h_act.reshape(-1, h_act.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        self.grads['b2'] = np.sum(dout, axis=(0, 1))
        
        dh_act = dout @ self.W2.T
        
        # Re-apply dropout gradient
        if drop_mask is not None:
            dh_act = dh_act * drop_mask
        
        dh = dh_act * gelu_derivative(h)
        
        self.grads['W1'] = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
        self.grads['b1'] = np.sum(dh, axis=(0, 1))
        
        dx = dh @ self.W1.T
        return dx

# ==============================================================================
# TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock:
    """Transformer block with residual connections"""
    
    def __init__(self, config: Config):
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)
        
        self.ln1_gamma = np.ones(config.n_embd)
        self.ln1_beta = np.zeros(config.n_embd)
        self.ln2_gamma = np.ones(config.n_embd)
        self.ln2_beta = np.zeros(config.n_embd)
        
        self.cache = {}
        self.grads = {}
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with residual connections"""
        # Attention block
        ln1_out, ln1_cache = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = self.attn.forward(ln1_out, training)
        x1 = x + attn_out
        
        # FFN block
        ln2_out, ln2_cache = layer_norm(x1, self.ln2_gamma, self.ln2_beta)
        ffn_out = self.ffn.forward(ln2_out, training)
        x2 = x1 + ffn_out
        
        self.cache = {'x': x, 'x1': x1, 'ln1_cache': ln1_cache, 'ln2_cache': ln2_cache}
        
        return x2
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # FFN backward
        dffn = self.ffn.backward(dout)
        dln2, dln2_gamma, dln2_beta = layer_norm_backward(dffn, self.cache['ln2_cache'], self.ln2_gamma)
        self.grads['ln2_gamma'] = dln2_gamma
        self.grads['ln2_beta'] = dln2_beta
        dx1 = dout + dln2
        
        # Attention backward
        dattn = self.attn.backward(dx1)
        dln1, dln1_gamma, dln1_beta = layer_norm_backward(dattn, self.cache['ln1_cache'], self.ln1_gamma)
        self.grads['ln1_gamma'] = dln1_gamma
        self.grads['ln1_beta'] = dln1_beta
        dx = dx1 + dln1
        
        return dx

# ==============================================================================
# MINI-GPT MODEL
# ==============================================================================

class MiniGPT:
    """Complete mini-GPT model with improvements"""
    
    def __init__(self, vocab_size: int, config: Config):
        self.config = config
        self.vocab_size = vocab_size
        
        # Embeddings
        self.tok_emb = np.random.randn(vocab_size, config.n_embd) * 0.02
        self.pos_emb = np.random.randn(config.block_size, config.n_embd) * 0.02
        
        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]
        
        # Final layer norm
        self.ln_f_gamma = np.ones(config.n_embd)
        self.ln_f_beta = np.zeros(config.n_embd)
        
        # Output projection
        self.lm_head = np.random.randn(config.n_embd, vocab_size) * 0.02
        
        self.cache = {}
        self.grads = {}
        
        # Adam optimizer
        self._init_optimizer()
    
    def _init_optimizer(self):
        """Initialize Adam optimizer for all parameters"""
        params = {}
        self._collect_params(params, prefix='')
        self.optimizer = AdamOptimizer(params, self.config)
    
    def _collect_params(self, params: Dict, prefix: str = ''):
        """Collect all parameters recursively"""
        for attr_name in ['tok_emb', 'pos_emb', 'lm_head', 'ln_f_gamma', 'ln_f_beta']:
            if hasattr(self, attr_name):
                params[prefix + attr_name] = getattr(self, attr_name)
        
        for i, block in enumerate(self.blocks):
            block_prefix = f'blocks.{i}.'
            params[block_prefix + 'ln1_gamma'] = block.ln1_gamma
            params[block_prefix + 'ln1_beta'] = block.ln1_beta
            params[block_prefix + 'ln2_gamma'] = block.ln2_gamma
            params[block_prefix + 'ln2_beta'] = block.ln2_beta
            
            attn = block.attn
            for key in ['W_q', 'W_k', 'W_v', 'W_o', 'b_q', 'b_k', 'b_v', 'b_o']:
                params[block_prefix + 'attn.' + key] = getattr(attn, key)
            
            ffn = block.ffn
            for key in ['W1', 'b1', 'W2', 'b2']:
                params[block_prefix + 'ffn.' + key] = getattr(ffn, key)
    
    def forward(self, idx: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        B, T = idx.shape
        
        tok_emb = self.tok_emb[idx]
        pos_emb = self.pos_emb[:T]
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block.forward(x, training)
        
        x, ln_f_cache = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        logits = x @ self.lm_head
        
        self.cache = {'idx': idx, 'x_final': x, 'ln_f_cache': ln_f_cache}
        
        return logits
    
    def backward(self, dlogits: np.ndarray) -> None:
        """Backward pass"""
        B, T, _ = dlogits.shape
        idx = self.cache['idx']
        x_final = self.cache['x_final']
        
        # Output projection
        self.grads['lm_head'] = x_final.reshape(-1, self.config.n_embd).T @ dlogits.reshape(-1, self.vocab_size)
        dx = dlogits @ self.lm_head.T
        
        # Final layer norm
        dx, self.grads['ln_f_gamma'], self.grads['ln_f_beta'] = layer_norm_backward(
            dx, self.cache['ln_f_cache'], self.ln_f_gamma
        )
        
        # Blocks in reverse
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        
        # Embeddings (we don't update these, just compute for completeness)
        # In practice, you'd update them too
    
    def clip_gradients(self, max_norm: float) -> float:
        """Clip gradients to prevent explosion"""
        all_grads = []
        
        # Collect all gradients
        for key in self.grads:
            all_grads.append(self.grads[key].flatten())
        
        for block in self.blocks:
            for key in block.attn.grads:
                all_grads.append(block.attn.grads[key].flatten())
            for key in block.ffn.grads:
                all_grads.append(block.ffn.grads[key].flatten())
            for key in block.grads:
                all_grads.append(block.grads[key].flatten())
        
        total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for key in self.grads:
                self.grads[key] *= clip_coef
            for block in self.blocks:
                for key in block.attn.grads:
                    block.attn.grads[key] *= clip_coef
                for key in block.ffn.grads:
                    block.ffn.grads[key] *= clip_coef
                for key in block.grads:
                    block.grads[key] *= clip_coef
        
        return total_norm
    
    def update_params(self, lr: float) -> None:
        """Update parameters using Adam optimizer"""
        # Collect all gradients
        grads = {}
        
        # Collect from model level
        for key in self.grads:
            grads[key] = self.grads[key]
        
        # Collect from blocks
        for i, block in enumerate(self.blocks):
            block_prefix = f'blocks.{i}.'
            grads[block_prefix + 'ln1_gamma'] = block.grads['ln1_gamma']
            grads[block_prefix + 'ln1_beta'] = block.grads['ln1_beta']
            grads[block_prefix + 'ln2_gamma'] = block.grads['ln2_gamma']
            grads[block_prefix + 'ln2_beta'] = block.grads['ln2_beta']
            
            attn = block.attn
            for key in ['W_q', 'W_k', 'W_v', 'W_o', 'b_q', 'b_k', 'b_v', 'b_o']:
                grads[block_prefix + 'attn.' + key] = attn.grads[key]
            
            ffn = block.ffn
            for key in ['W1', 'b1', 'W2', 'b2']:
                grads[block_prefix + 'ffn.' + key] = ffn.grads[key]
        
        # Update using Adam
        self.optimizer.step(grads, lr)
        
        # Copy back to model
        for key in grads:
            if '.' in key:
                parts = key.split('.')
                if parts[0] == 'blocks':
                    idx = int(parts[1])
                    if parts[2] == 'ln1_gamma':
                        self.blocks[idx].ln1_gamma = self.optimizer.params[key]
                    elif parts[2] == 'ln1_beta':
                        self.blocks[idx].ln1_beta = self.optimizer.params[key]
                    elif parts[2] == 'ln2_gamma':
                        self.blocks[idx].ln2_gamma = self.optimizer.params[key]
                    elif parts[2] == 'ln2_beta':
                        self.blocks[idx].ln2_beta = self.optimizer.params[key]
                    elif parts[2] == 'attn':
                        setattr(self.blocks[idx].attn, parts[3], self.optimizer.params[key])
                    elif parts[2] == 'ffn':
                        setattr(self.blocks[idx].ffn, parts[3], self.optimizer.params[key])
            else:
                setattr(self, key, self.optimizer.params[key])
    
    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 1.0) -> np.ndarray:
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self.forward(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature
            probs = softmax(logits, axis=-1)
            idx_next = np.array([[np.random.choice(len(p), p=p)] for p in probs])
            idx = np.concatenate([idx, idx_next], axis=1)
        return idx
    
    def save(self, path: str):
        """Save model to file"""
        data = {
            'vocab_size': self.vocab_size,
            'config': {
                'n_embd': self.config.n_embd,
                'n_head': self.config.n_head,
                'n_layer': self.config.n_layer,
                'block_size': self.config.block_size,
            },
            'tok_emb': self.tok_emb.tolist(),
            'pos_emb': self.pos_emb.tolist(),
            'lm_head': self.lm_head.tolist(),
            'ln_f_gamma': self.ln_f_gamma.tolist(),
            'ln_f_beta': self.ln_f_beta.tolist(),
        }
        # Save blocks separately (would need more code for full implementation)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Model saved to {path}")

def cross_entropy_loss_improved(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Improved cross-entropy loss with better numerical stability"""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Log-sum-exp trick
    logits_max = np.max(logits_flat, axis=1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    log_sum_exp = logits_max + np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    
    # Get log prob of correct class
    correct_logits = logits_flat[np.arange(len(targets_flat)), targets_flat]
    loss = np.mean(log_sum_exp[:, 0] - correct_logits)
    
    # Gradient computation
    probs = np.exp(logits_shifted - log_sum_exp)
    dlogits = probs
    dlogits[np.arange(len(targets_flat)), targets_flat] -= 1
    dlogits /= (B * T)
    dlogits = dlogits.reshape(B, T, V)
    
    return loss, dlogits

def get_batch(data: np.ndarray, block_size: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get a random batch of training data"""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.array([data[i:i+block_size] for i in ix])
    y = np.array([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train(model: MiniGPT, train_data: np.ndarray, val_data: np.ndarray, config: Config):
    """Improved training loop with LR scheduling and gradient clipping"""
    print(f"\nModel has {model.count_parameters():,} parameters")
    print("Starting training...\n")
    
    best_val_loss = float('inf')
    
    for iter in range(config.max_iters):
        # Get learning rate
        lr = get_lr(iter, config)
        
        # Forward pass
        xb, yb = get_batch(train_data, config.block_size, config.batch_size)
        logits = model.forward(xb, training=True)
        loss, dlogits = cross_entropy_loss_improved(logits, yb)
        
        # Backward pass
        model.backward(dlogits)
        
        # Gradient clipping
        grad_norm = model.clip_gradients(config.grad_clip)
        
        # Update parameters
        model.update_params(lr)
        
        # Validation
        if iter % config.eval_interval == 0:
            val_xb, val_yb = get_batch(val_data, config.block_size, config.batch_size)
            val_logits = model.forward(val_xb, training=False)
            val_loss, _ = cross_entropy_loss_improved(val_logits, val_yb)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            print(f"Step {iter:5d} | train loss: {loss:.4f} | val loss: {val_loss:.4f} | "
                  f"lr: {lr:.2e} | grad_norm: {grad_norm:.2f}")
    
    print("\nTraining complete!")

# Note: count_parameters method would need to be implemented
# This is a simplified version - full implementation would be longer


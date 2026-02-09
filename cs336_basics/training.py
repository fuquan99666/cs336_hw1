# in this training file ,we will implement the training component of our transformer model

from jaxtyping import Float,Int
from typing import Optional, Callable, Iterable
from torch import Tensor
from cs336_basics.transformer import softmax
import torch
import math
import numpy as np
import time
import os
import csv

def cross_entropy(
        inputs: Float[Tensor, "... seq_len vocab_size"], 
        targets: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, ""]:

    seq_len = inputs.shape[-2]
    
    # # 1. 数值稳定版本的log-softmax
    # # 减去最大值确保数值稳定
    max_vals = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - max_vals
    
    # # 2. 计算log-softmax: log(exp(x_i)/sum(exp(x_j)))
    # #    = x_i - log(sum(exp(x_j))) !!!
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_stable), dim=-1, keepdim=True))
    log_probs = inputs_stable - log_sum_exp  # [seq_len, vocab_size]
    # print(log_probs.shape)
    # print(targets.shape)
    # print(torch.arange(seq_len).shape)
    
    # # 3. 选择目标词对应的对数概率
    #target_log_probs = log_probs[torch.arange(seq_len), targets]  # [seq_len]
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
 
    # # 4. 计算负对数似然并平均
    loss = -target_log_probs.mean()
    
    return loss

# implement a adamW  optimizer

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr,betas,eps,weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                grad2 = grad * grad # Element-wise square of the gradient.
                m = (1-beta1) * grad + beta1 * state.get("m", torch.zeros_like(p.grad.data))
                v = (1-beta2) * grad2 + beta2 * state.get("v", torch.zeros_like(p.grad.data))
                lr_t = lr * (math.sqrt(1 - beta2 ** (t)) / (1 - beta1 ** (t)))
                p.data -=  lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return 
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # based it,max,min,wi,cci
    if it < warmup_iters:
        lr = max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate 
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    total_norm = 0.0
    for p in parameters:
        grad = p.grad
        if grad is None:
            continue
        p2 = grad.data.pow(2).sum()
        total_norm += p2.item()
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        k = max_l2_norm / (total_norm + eps)
        for p in parameters:  
            if p.grad is None:
                continue  
            p.grad *= k

    return

import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # return a tuple (inputs,targets)
    num_tokens = dataset.shape[0]
    inputs = torch.zeros((batch_size, context_length), device=device, dtype=torch.long)
    targets = torch.ones((batch_size, context_length), device=device, dtype=torch.long)

    for i in range(batch_size):
        start = torch.randint(num_tokens - context_length,(1,)).item()
        end = (start + context_length) # actually add more 1, but think about we will use target, magic!
        if end < num_tokens: # if end < start, ignore this case
            inputs[i] = torch.tensor(dataset[start:end], device=device, dtype=torch.long)
            targets[i] = torch.tensor(dataset[start+1:end+1], device=device, dtype=torch.long)

    return (inputs, targets)

@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, max_batches=32):
    model.eval()
    losses = []

    for _ in range(max_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(model, optimizer, iteration, out):
    # return : int: the previously-serialized number of iterations.
    # we use a dictionary to save the model and optimizer state_dict
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    return iteration

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

def run_train(args_list=None):
    """训练一个完整的transformer语言模型
    
    功能：
    1. 支持自定义模型和优化器超参数
    2. 使用np.memmap高效加载大数据集
    3. 支持checkpoint保存和恢复
    4. 定期打印训练loss
    """
    import argparse
    from cs336_basics.transformer import TransformerLM
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    
    # 模型超参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    
    # 优化器超参数
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_learning_rate', type=float, default=6e-4)
    parser.add_argument('--min_learning_rate', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # 学习率调度超参数
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000)
    
    # 训练超参数
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)
    
    # 数据和checkpoint路径
    parser.add_argument('--dataset_path', type=str,default="./data/tiny_train_bpe_token_ids.npy")
    parser.add_argument('--val_dataset_path', type=str,default="./data/tiny_valid_bpe_token_ids.npy")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth')
    parser.add_argument('--resume_from', type=str, default=None, help='恢复训练的checkpoint路径')
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save experiment logs')
    parser.add_argument('--exp_name', type=str, default='train_tiny_valid', help='Name of the experiment (for log filename)')
    
    args = parser.parse_args(args_list)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # 加载数据集
    print(f'Loading dataset from {args.dataset_path}...')
    if args.dataset_path.endswith('.npy'):
        dataset = np.load(args.dataset_path, mmap_mode='r')
        print(f"Successfully loaded .npy dataset!")
    else:
        dataset = np.memmap(args.dataset_path, dtype=np.uint16, mode='r')
    print(f'Dataset size: {len(dataset)} tokens')
    
    # Load validation dataset if provided
    val_dataset = None
    if args.val_dataset_path:
        print(f'Loading validation dataset from {args.val_dataset_path}...')
        if args.val_dataset_path.endswith('.npy'):
            val_dataset = np.load(args.val_dataset_path, mmap_mode='r')
        else:
            val_dataset = np.memmap(args.val_dataset_path, dtype=np.uint16, mode='r')
        print(f'Val Dataset size: {len(val_dataset)} tokens')
    
    # 初始化模型
    print('Initializing model...')
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {num_params:,} parameters')
    
    # 初始化优化器
    # 修复：将参数分为两组，一组应用 weight_decay，一组（如LayerNorm和bias）不应用
    # 虽然现在的模型Linear没有bias，但这是一种通用的最佳实践
    optim_groups = [
        # 对 weight (如果是2D或更高维，通常是 Linear/Embedding 的权重) 应用 decay
        {'params': [p for n, p in model.named_parameters() if p.dim() >= 2], 'weight_decay': args.weight_decay},
        # 对 bias 和 1D 参数 (如 LayerNorm/RMSNorm 的权重) 不应用 decay
        # 很实际的一件事，矩阵参数我们不希望某个参数占比过大，所以应用weight decay，但是对于layernorm，其实无所谓
        {'params': [p for n, p in model.named_parameters() if p.dim() < 2], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        params=optim_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    log_path = os.path.join(args.log_dir, f"{args.exp_name}.csv")
    print(f"Logging metrics to {log_path}")

    # 恢复训练（如果指定）
    start_iter = 0
    if args.resume_from:
        print(f'Resuming from checkpoint: {args.resume_from}')
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        log_file = open(log_path, 'a', newline='')
        print(f'Resumed from iteration {start_iter}')
    else:
        log_file = open(log_path, 'w', newline='')
    # Setup Logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    

    
    # Open CSV file for logging
    # Use 'a' (append) mode to support resuming, or 'w' to overwrite. 
    # If resuming, we ideally want to append. If it's a new run with same name, maybe overwrite?
    # Let's assume distinct names or append behavior.
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    csv_writer = csv.writer(log_file)
    
    if not file_exists:
        csv_writer.writerow(['iteration', 'loss', 'val_loss', 'lr', 'wallclock_time'])

    start_time = time.time()
    # Note: Wallclock time tracking when resuming is tricky. 
    # We will log "time since this script started". Data analysis can handle offsets if needed.

    # 训练循环
    print('Starting training...')
    model.train()
    
    for iteration in range(start_iter, args.max_iters):
        # 动态调整学习率
        lr = get_lr_cosine_schedule(
            iteration,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 获取一批数据
        inputs, targets = get_batch(
            dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device
        )
        
        # 前向传播
        logits = model(inputs)
        
        # 计算损失
        loss = cross_entropy(logits, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # 优化器步进
        optimizer.step()
        
        # 打印日志
        if iteration % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            val_loss = 0.0
            if val_dataset is not None:
                val_loss = evaluate(model, val_dataset, args.batch_size, args.context_length, device)
            
            #print(f'Iter {iteration:5d} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f} | Time: {elapsed_time:.2f}s')
            csv_writer.writerow([iteration, loss.item(), val_loss, lr, elapsed_time])
            log_file.flush()
        
        # 保存checkpoint
        # if iteration % args.save_interval == 0 and iteration > 0:
        #     print(f'Saving checkpoint at iteration {iteration}...')
        #     save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
    
    # 训练结束，保存最终模型
    log_file.close()
    print(f'Training complete. Saving final checkpoint...')
    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
    print('Done!')


def generate():
    """
    Generate completions for a user-provided prompt (i.e., take in some x1...t and sample a completion
    until you hit an <|endoftext|> token).

    Allow the user to control the maximum number of generated tokens.

    Given a desired temperature value, apply softmax temperature scaling to the predicted next-word
    distributions before sampling.

    Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a user-specified
    threshold value.
    """
    import argparse
    import torch
    import json
    from cs336_basics.transformer import TransformerLM
    from cs336_basics.bpe_tokenizer import BPETokenizer
    
    parser = argparse.ArgumentParser(description='Generate text from a trained language model')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/tiny_lr_3.0e-03_with_batchsize_128.pth", help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default="./data/bpe_train_tiny_vocab.json", help='Path to vocabulary JSON file')
    parser.add_argument('--merges', type=str, default="./data/bpe_train_tiny_merges.json", help='Path to merges file')
    
    # Model architecture parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for model parameters')
    # parser.add_argument('--attn_pdrop', type=float, default=0.1, help='Attention dropout')
    # parser.add_argument('--residual_pdrop', type=float, default=0.1, help='Residual dropout')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='./input.txt', help='Input prompt to continue from')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    
    # 加载tokenizer
    print('Loading tokenizer...')
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )
    
    # 创建模型
    print('Creating model...')
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta = args.rope_theta,
        device=device,
        dtype=dtype
    )
    
    # 加载模型权重
    print('Loading checkpoint...')
    # 此时我们不需要进行训练，所以没必要加载optimizer状态，所以直接用torch.load加载模型权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    if args.prompt:
        with open(args.prompt, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        input_ids = tokenizer.encode(prompt_text)
        print(f'input_ids: {input_ids}')
        print(f'Prompt: {prompt_text}')
    else:
        input_ids = []
        print('Starting with empty prompt')
    
    # 获取<|endoftext|>的token id
    endoftext_token = tokenizer.vocab_reverse.get(b'<|endoftext|>', None)
    
    print(f'\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})...\n')
    
    # 生成tokens
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(args.max_tokens):
            # 准备输入（取最后context_length个tokens）
            context = generated_ids[-args.context_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            # ---------------- 修改开始 ----------------
            # 计算正确的 token_positions
            # 1. 知道当前生成的总长度
            current_total_len = len(generated_ids)
            # 2. 知道窗口起始位置
            start_pos = max(0, current_total_len - args.context_length)
            # 3. 知道窗口相对长度 (len(context))
            window_len = len(context)
            # 4. 生成绝对位置索引： [start_pos, start_pos+1, ..., start_pos + window_len - 1]
            token_positions = torch.arange(start_pos, start_pos + window_len, device=device)
            
            # 前向传播，传入位置
            logits = model(x, token_positions=token_positions)  # (1, seq_len, vocab_size)
            # ---------------- 修改结束 ----------------
            
            logits = logits[0, -1, :]  # 取最后一个位置的logits
            
            # Temperature scaling
            if args.temperature != 1.0:
                logits = logits / args.temperature
            
            # 转换为概率
            probs = torch.softmax(logits, dim=-1)
            
            # Top-p (nucleus) sampling
            if args.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > args.top_p
                # 保留第一个超过阈值的token（确保至少有一个token）
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                # 将不在nucleus中的token概率设为0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                
                # 重新归一化
                probs = probs / probs.sum()
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token)
            
            # 检查是否遇到<|endoftext|>
            if endoftext_token is not None and next_token == endoftext_token:
                print('Reached <|endoftext|> token')
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids)
    print('Generated text:')
    print('-' * 80)
    print(generated_text)
    print('-' * 80)
    
    return generated_text



from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
def test():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

if __name__ == "__main__":
    #test()
    generate()
    #run_train()
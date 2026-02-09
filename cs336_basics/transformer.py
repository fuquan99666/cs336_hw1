# implement a Linear class that herit fromo torch.Module
import torch
import einops

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # use torch.nn.init.trunc_normal_ to initialize weight, and we don't need bias
        # and put W in nn.Parameter
        self.W = torch.nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # realize the linear transformation y = xW^T
        return torch.matmul(x, self.W)
        


# implement a embedding class that herit from torch.Module
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # we will create a num_embeddings x embedding_dim weight matrix
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # initialize the weight matrix
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # select just the index of token_ids from weight matrix
        # I must say this is a tricky expression of tensor indexing in pytorch
        return self.weight[token_ids]


# implement a pre_norm  RMSNorm class that herit from torch.Module
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # create a weight parameter of size d_model as g_i
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) 
        x_norm = x * self.weight / rms
        return x_norm.to(in_dtype)

# implement a swiglu ffn class that herit from torch.Module
class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        # maybe ?
        if d_ff is None:
            d_ff = 8 / 3 * d_model
            # make it 64 aligned
            d_ff = (d_ff + 63) // 64 * 64
        else:
            d_ff = (d_ff + 63) // 64 * 64
        self.d_ff = d_ff
    
        self.device = device
        self.dtype = dtype
        # create w1,w2,w3
        self.w1 = torch.nn.Parameter(
            torch.empty(self.d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(d_model, self.d_ff, device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(self.d_ff,d_model, device=device, dtype=dtype)
        )
        
        # init
        std = (2 / (d_model + self.d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use einop to implement the swiglu ffn
        x1 = einops.einsum(x,self.w1,'... d_model, d_ff d_model -> ... d_ff')
        # use SiLU activation
        x1 = x1 * torch.sigmoid(x1)
        x2 = einops.einsum(x,self.w3,'... d_model, d_ff d_model -> ... d_ff')
        # use gate dot product
        # x3 = einops.einsum(x1,x2,'... d_ff, ... d_ff -> ... d_ff')
        x3 = x1 * x2
        x3 = einops.einsum(x3,self.w2,'... d_ff, d_model d_ff -> ... d_model')
        return x3
    
# implement a RoPE positional encoding class
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0


        self.d_k = d_k
        self.max_seq_len = max_seq_len


        # precompute frequencies
        # 强制使用 float32 计算频率和角度，防止精度溢出导致位置编码失效
        dim = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim / d_k))


        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->i j", positions, inv_freq)


        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        # x shape is (... ,seq_len, d_k)
        # token_positions shape is (..., seq_len)
        # we need to expand the cos and sin to match the shape of x and fit the token_positions

        cos = self.cos[token_positions] # shape (..., seq_len, d_k//2)
        sin = self.sin[token_positions] # shape (..., seq_len, d_k//2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]


        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos


        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd


        return x_rot
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # input is a tensor, and a fixed dimension 
    # we should do softmax along that dim and use sub_max trick for numerical stability

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum 


def SiLU(x: torch.Tensor) -> torch.Tensor:
    # input a tensor , return a tensor after SiLU activation
    return x * torch.sigmoid(x)


class FFNSiLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # create w1,w2
        self.w1 = torch.nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )

        # init
        std = (2 / (d_model + d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einops.einsum(x,self.w1,'... d_model, d_ff d_model -> ... d_ff')
        x1 = SiLU(x1)
        x2 = einops.einsum(x1,self.w2,'... d_ff, d_model d_ff -> ... d_model')
        return x2


def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.tensor = None
) -> torch.Tensor:
    # realize the scaled dot product attention
    d_k = Q.shape[-1]
    scores = torch.einsum('... i j, ... k j -> ... i k', Q, K) / (d_k ** 0.5)
    if mask is not None:
        scores = torch.where(mask == False, float('-inf'), scores)
    attention_scores = softmax(scores,dim=-1)
    ouput = torch.einsum('... i k, ... k v -> ... i v', attention_scores, V)
    return ouput


class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int,seq_len: int ,num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.length = seq_len
        self.d_v = d_model // num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.dtype = dtype

        # q,k,v,o as parameters
        self.q = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
        )
        self.k = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
        )
        self.v = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
        )
        self.o = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
        )
        self.register_buffer("mask", torch.tril(torch.ones((2048, 2048), device=self.device, dtype=torch.bool)), persistent=False)

        std = (2 / (d_model + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.q, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.k, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.v, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.o, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor, rope: RoPE = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        # the input q,k,v is three weights
        # first we need to compute Q,K,V from q,k,v weights * input x
        # Q = torch.einsum('... i j,k j -> ... i k', x,self.q )
        # K = torch.einsum('... i j,k j -> ... i k', x,self.k )
        # V = torch.einsum('... i j,k j -> ... i k', x,self.v )

        qkv_weight = torch.cat([self.q, self.k, self.v], dim=0)  # shape (3*d_model, d_model)
        # 然后进行一次矩阵乘法
        qkv = einops.einsum(x, qkv_weight, '... len d_model, three d_model -> ... len three')
        # 最后再拆分成Q,K,V
        #Q, K, V = torch.split(qkv, self.d_model, dim=-1)  # each shape (..., len, d_model)


        # Q = einops.rearrange(Q, '... l (n k) -> ... n l k', n=self.num_heads)
        # K = einops.rearrange(K, '... l (n k) -> ... n l k', n=self.num_heads)
        # V = einops.rearrange(V, '... l (n v) -> ... n l v', n=self.num_heads)

        qkv = einops.rearrange(
                qkv,
                '... l (three n k) -> ... n l (three k)',
                n=self.num_heads,
                k=self.d_k 
        )

        Q,K,V = torch.split(qkv, self.d_k, dim=-1)
        # Q, K, V = qkv

        # safe check for mask
        length = Q.shape[-2]
        if length > self.mask.shape[0]:
            mask = torch.tril(torch.ones((length, length), device=self.device, dtype=torch.bool))
        else:
            mask = self.mask[:length, :length]

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            Q = rope.forward(Q, token_positions)
            K = rope.forward(K, token_positions)
        length = Q.shape[-2]
        mask = self.mask[:length, :length]

        attention = scaled_dot_product_attention(Q,K,V,mask=mask)

        # concat the multi head output
        attention = einops.rearrange(attention, '... n l v -> ... l (n v)' )

        # finally project the output with o weight
        # notice that here we use o's shape is (d_model,n*hv) to do projection
        # n*hv <-> n*v 
        output = einops.einsum(attention, self.o,'... len hd,d_model hd -> ... len d_model')
        return output
    


    def forward_with_rope(
            self, x: torch.tensor, token_positions: torch.tensor, rope: RoPE
        ) -> torch.tensor:
        # Q = torch.einsum('... i j,k j -> ... i k', x,self.q )
        # K = torch.einsum('... i j,k j -> ... i k', x,self.k )
        # V = torch.einsum('... i j,k j -> ... i k', x,self.v )       

        # 优化一下三个矩阵的乘法
        # 先合并三个weight矩阵
        qkv_weight = torch.cat([self.q, self.k, self.v], dim=0)  # shape (3*d_model, d_model)
        # 然后进行一次矩阵乘法
        qkv = einops.einsum(x, qkv_weight, '... len d_model, three d_model -> ... len three')
        # 最后再拆分成Q,K,V
        #Q, K, V = torch.split(qkv, self.d_model, dim=-1)  # each shape (..., len, d_model)


        # Q = einops.rearrange(Q, '... l (n k) -> ... n l k', n=self.num_heads)
        # K = einops.rearrange(K, '... l (n k) -> ... n l k', n=self.num_heads)
        # V = einops.rearrange(V, '... l (n v) -> ... n l v', n=self.num_heads)

        qkv = einops.rearrange(
                qkv,
                '... l (three n k) -> ... n l (three k)',
                n=self.num_heads,
                k=self.d_k 
        )

        Q,K,V = torch.split(qkv, self.d_k, dim=-1)
        # Q, K, V = qkv

        length = Q.shape[-2]
        mask = torch.tril(torch.ones((length, length), device=self.device, dtype=torch.bool))


        Q = rope.forward(Q, token_positions)
        K = rope.forward(K, token_positions)

        attention = scaled_dot_product_attention(Q,K,V,mask=mask)
        attention = einops.rearrange(attention, '... n l v -> ... l (n v)' )
        output = einops.einsum(attention, self.o,'... len hd,d_model hd -> ... len d_model')
        return output


 
from jaxtyping import Float,Int
from torch import Tensor


class TransformerBlock(torch.nn.Module):
    """Transformer解码器块，包含自注意力和前馈网络"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seq_len: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        
        # 层归一化
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 多头自注意力
        self.attn = multihead_self_attention(d_model,seq_len ,num_heads, device=device, dtype=dtype)
        
        # 前馈网络
        self.ffn = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)
        #self.ffn = FFNSiLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"],
        rope: RoPE,
        token_positions: Tensor = None
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        # 第一个子层：自注意力 + 残差连接 (pre-norm)
        x_norm1 = self.ln1(x)


        # 使用RoPE的自注意力
        #token_positions = torch.arange(x.shape[-2], device=x.device)
        attn_output = self.attn(
            x_norm1, 
            token_positions=token_positions, 
            rope=rope
        )
        # attn_output = self.attn.forward_with_rope(
        #     x_norm1, 
        #     token_positions=torch.arange(x.shape[-2], device=x.device),
        #     rope=rope
        # )
        x = x + attn_output
        
        # 第二个子层：前馈网络 + 残差连接 (pre-norm)
        x_norm2 = self.ln2(x)

        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output

        
        return x



class TransformerLM(torch.nn.Module):
    """完整的Transformer语言模型"""
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embedding
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # rope
        # 修正：RoPE的max_seq_len应该设置得足够大，以支持推理时生成比context_length更长的文本
        # 这里设置为 max(context_length, 4096)，保证至少能处理 4096 长度的绝对位置
        rope_max_len = max(context_length, 4096)
        self.rope = RoPE(rope_theta, d_model // num_heads, rope_max_len, device=device)

        # Transformer blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                seq_len = context_length,
                
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Language model head
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
        token_positions: Tensor = None 
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        # Token embedding
        x = self.token_embeddings(in_indices)
        
        # 通过所有transformer blocks
        for layer in self.layers:
            x = layer(x,self.rope, token_positions=token_positions)
        
        # Final normalization
        x = self.ln_final(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        return logits


# 保留函数式接口以兼容测试
def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """函数式transformer block接口，用于测试
    
    这个函数从weights字典中加载预训练权重，而不是训练模型。
    """
    # 第一个子层：pre-norm + 自注意力 + 残差
    rms1 = RMSNorm(d_model, device=in_features.device, dtype=in_features.dtype)
    rms1.load_state_dict({
        'weight': weights['ln1.weight']
    })
    x_norm1 = rms1.forward(in_features)

    mha = multihead_self_attention(d_model, num_heads, device=in_features.device, dtype=in_features.dtype)
    mha.load_state_dict({
        'q': weights['attn.q_proj.weight'],
        'k': weights['attn.k_proj.weight'],
        'v': weights['attn.v_proj.weight'],
        'o': weights['attn.output_proj.weight'],
    })

    token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
    atten_output = mha.forward_with_rope(x_norm1, token_positions=token_positions, max_seq_len=max_seq_len, theta=theta)
    x1 = in_features + atten_output

    # 第二个子层：pre-norm + FFN + 残差
    rms2 = RMSNorm(d_model, device=in_features.device, dtype=in_features.dtype)
    rms2.load_state_dict({
        'weight': weights['ln2.weight']
    })
    x_norm2 = rms2.forward(x1)  

    ffn = SwiGLUFFN(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    ffn.load_state_dict({
        'w1': weights['ffn.w1.weight'],
        'w2': weights['ffn.w2.weight'],
        'w3': weights['ffn.w3.weight'],
    })

    ffn_output = ffn.forward(x_norm2)
    x2 = x1 + ffn_output

    return x2


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """函数式transformer LM接口，用于测试
    
    这个函数从weights字典中加载预训练权重，而不是训练模型。
    """
    # Token embedding
    embedding = Embedding(vocab_size, d_model, device=in_indices.device, dtype=torch.float32)
    embedding.load_state_dict({
        'weight': weights['token_embeddings.weight']
    })
    x = embedding.forward(in_indices)

    # 通过所有transformer blocks
    for layer_idx in range(num_layers):
        x = transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            {
                'ln1.weight': weights[f'layers.{layer_idx}.ln1.weight'],
                'attn.q_proj.weight': weights[f'layers.{layer_idx}.attn.q_proj.weight'],
                'attn.k_proj.weight': weights[f'layers.{layer_idx}.attn.k_proj.weight'],
                'attn.v_proj.weight': weights[f'layers.{layer_idx}.attn.v_proj.weight'],
                'attn.output_proj.weight': weights[f'layers.{layer_idx}.attn.output_proj.weight'],
                'ln2.weight': weights[f'layers.{layer_idx}.ln2.weight'],
                'ffn.w1.weight': weights[f'layers.{layer_idx}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{layer_idx}.ffn.w2.weight'],
                'ffn.w3.weight': weights[f'layers.{layer_idx}.ffn.w3.weight'],
            },
            x
        )
    
    # Final normalization
    final_rms = RMSNorm(d_model, device=in_indices.device, dtype=torch.float32)
    final_rms.load_state_dict({
        'weight': weights['ln_final.weight']
    })
    x_norm = final_rms.forward(x)

    # Language model head
    final_linear = Linear(d_model, vocab_size, device=in_indices.device, dtype=torch.float32)
    final_linear.load_state_dict({
        'W': weights['lm_head.weight'].T
    })
    logits = final_linear.forward(x_norm)

    return logits

import torch 
import torch.nn as nn

"""MODEL"""

import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.num_channels = config["num_channels"]
        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]
        
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        # print(x.shape)
        x = x.flatten(2).transpose(1,2)
        return x


class Embeddings(nn.Module):
    # Patch Embeddings + (CLS Token + Positional Embeddings )
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.embed_dim = config["embed_dim"]
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches+1, self.embed_dim))
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, attention_head_size,config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.attention_head_size = attention_head_size
        self.bias = config["bias"]
        
        self.query = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)
        self.key = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)
        self.value = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.matmul(q, k.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        attention_out = torch.matmul(attention_scores, v)
        
        return attention_out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.head_size = self.embed_dim//self.num_heads
        self.all_head_size = self.head_size * self.num_heads
        self.dropout = config["dropout"]
        self.qkv_bias = config["bias"]

        self.heads = nn.ModuleList([
            AttentionHead(
                self.head_size,
                config
            ) for _ in range(self.num_heads)
        ])

        self.attention_mlp = nn.Linear(self.all_head_size, self.embed_dim)
        self.out_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output in attention_outputs], dim=-1) #concat attention for each head
        attention_output = self.attention_mlp(attention_output)
        attention_output = self.out_dropout(attention_output)

        return attention_output
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
        # self.act = nn.GELU()
        self.act=NewGELUActivation()
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.dropout=nn.Dropout(config["dropout"])

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim=config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.hidden_dim = config["hidden_dim"]
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self,x):
        attention_output = self.attention(self.layer_norm1(x))
        x = x+attention_output
        mlp_out = self.mlp(self.layer_norm2(x))
        x = x+mlp_out
        return x
        
    
class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])
    def forward(self,x ):
        all_attentions = []
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.embed_dim = config["embed_dim"]
        self.num_classes = config["num_classes"]
    
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
    
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoder_output = self.encoder(embedding_output)
        classification = self.classifier(encoder_output[:, 0, :])
        return classification


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.positional_embeddings.data = nn.init.trunc_normal_(
                module.positional_embeddings.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.positional_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.cls_token.dtype)

            

import torch.nn as nn
import torch
import math
import numpy as np
from improved_diffusion.nn import timestep_embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.scale
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(RandomFourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1, embedding_dim // 2)) * scale, requires_grad=False)

    def forward(self, timesteps):
        emb = torch.mm(timesteps[:, None], self.w * 2 * 3.14159265359)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
    if embedding_type == 'positional':
        temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
    elif embedding_type == 'fourier':
        temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
    else:
        raise NotImplementedError

    return temb_fun


class Diffusion_Net(nn.Module):

    def __init__(self, model_opt):  #
        super(Diffusion_Net, self).__init__()
        # model_opt.learn_sigma = False
        model_opt.embedding_dim = model_opt.latent_size * model_opt.encoder_layers
        # model_opt.embedding_scale = 1000.0
        # model_opt.embedding_type = 'positional'
        self.predict_mlp = nn.Sequential(
            nn.Linear(model_opt.embedding_dim * 2, model_opt.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(model_opt.embedding_dim * 2, model_opt.embedding_dim * 2 if model_opt.learn_sigma else model_opt.embedding_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(model_opt.embedding_dim, model_opt.embedding_dim),
            nn.ReLU(),
            nn.Linear(model_opt.embedding_dim, model_opt.embedding_dim)
        )
        self.embedding_dim = model_opt.embedding_dim
        self.embedding_dim_mult = 4
        # self.temb_fun = init_temb_fun(model_opt.embedding_type, model_opt.embedding_scale, model_opt.embedding_dim)

    def forward(self, x, timesteps, y=None, w=0.1):
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(1)
        temb = self.time_mlp(timestep_embedding(timesteps, x.shape[-1]))
        if self.training and np.random.random() < 0.1:
            y = None
        if y is not None:
            temb += y

        noise = self.predict_mlp(torch.cat((x, temb), dim=-1))

        return noise
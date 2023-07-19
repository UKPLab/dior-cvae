import torch
import torch.nn as nn
#from NFmodels import AutoregressiveConditioner, AffineNormalizer, buildFCNormalizingFlow
# from .script_util import create_model

# This code shamely steals lines from Improved Denoising Diffusion Probabilistic Models

class DataDiffuser(nn.Module):
    def __init__(self, beta_min=1e-4, beta_max=.02, t_min=0, t_max=1000):
        super(DataDiffuser, self).__init__()

        self.register_buffer('betas', torch.linspace(beta_min, beta_max, t_max - t_min + 1))
        self.T = t_max
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', self.alphas.log().cumsum(0).exp())
        self.register_buffer('alphas_cumprod_prev', torch.cat((torch.tensor([1.]), self.alphas_cumprod[:-1]), 0))
        self.register_buffer('alphas_cumprod_next', torch.cat((self.alphas_cumprod[1:], torch.tensor([0.])), 0))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_variance', self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.register_buffer(
            'posterior_log_variance_clipped', torch.log(torch.cat((self.posterior_variance[[1]], self.posterior_variance[1:]), 0)))

        self.register_buffer(
            'posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.register_buffer(
            'posterior_mean_coef2', (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    def diffuse(self, x_t0, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        assert x_t0.shape[0] == t.shape[0]

        if noise is None:
            noise = torch.randn(x_t0.shape).to(x_t0.device)
        mu = self.sqrt_alphas_cumprod[t].view(-1, 1) * x_t0
        sigma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return mu + noise * sigma, (mu, sigma)

    def prev_mean_var(self, x_t, x_0, t):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)

        """

        mu = self.posterior_mean_coef1[t].view(-1, 1) * x_0 + self.posterior_mean_coef2[t].view(-1, 1) * x_t
        sigma = self.posterior_variance[t].view(-1, 1)
        return mu, sigma

    def past_sample(self, mu_z_pred, t_1, temperature=1.):
        log_sigma = self.posterior_log_variance_clipped[t_1].view(-1, 1)
        noise = torch.randn_like(mu_z_pred)
        return mu_z_pred + noise * torch.exp(.5 * log_sigma)


class AsynchronousDiffuser(nn.Module):
    def __init__(self, betas_min, betas_max, ts_min, ts_max, var_sizes):
        super(AsynchronousDiffuser, self).__init__()
        if not (len(betas_max) == len(betas_min) == len(ts_max) == len(ts_min) == len(var_sizes)):
            raise AssertionError

        t0 = 0
        T = max(ts_max)
        self.T = T
        betas = []
        alphas_t = []
        alphas = []
        dirac_z0 = []
        dirac_zt_1 = []
        for b_min, b_max, t_min, t_max, var_size in zip(betas_min, betas_max, ts_min, ts_max, var_sizes):
            beta = torch.zeros(var_size, T + 1)
            beta[:, t_min:t_max+1] = torch.linspace(b_min, b_max, 1 + t_max - t_min)

            betas.append(beta)

            dz0 = torch.zeros_like(beta)
            if t_min > 0:
                dz0[:, :t_min] = 1.

            dirac_z0.append(dz0)
            dzt1 = torch.zeros_like(beta)
            dzt1[:, t_max + 1:] = 1.
            dirac_zt_1.append(dzt1)

        self.register_buffer('betas', torch.cat(betas, 0).permute(1, 0).float())
        self.T = t_max
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', self.alphas.log().cumsum(0).exp())
        self.register_buffer('alphas_cumprod_prev', torch.cat((torch.tensor([1.]).view(1, 1).expand(1, sum(var_sizes)), self.alphas_cumprod[:-1, :]), 0))
        self.register_buffer('alphas_cumprod_next', torch.cat((self.alphas_cumprod[1:, :], torch.tensor([0.]).view(1, 1).expand(1, sum(var_sizes))), 0))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_variance', self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.cat((self.posterior_variance[[1]], self.posterior_variance[1:]), 0)))

        self.register_buffer(
            'posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))


        self.register_buffer('dirac_z0', torch.cat(dirac_z0, 0).permute(1, 0).float())
        self.register_buffer('dirac_zt_1', torch.cat(dirac_zt_1, 0).permute(1, 0).float())
        no_dirac = torch.ones_like(self.betas)
        no_dirac[self.dirac_z0 == 1.] = 0.
        no_dirac[self.dirac_zt_1 == 1.] = 0.
        self.register_buffer('no_dirac', no_dirac)

        #print(self.posterior_mean_coef1)
        self.posterior_mean_coef1[self.dirac_z0 == 1.] = 1.
        self.posterior_mean_coef2[self.dirac_z0 == 1.] = 0.
        #print(self.posterior_mean_coef1)
        #print((self.posterior_mean_coef2 + self.posterior_mean_coef1).max())
        #exit()

        # TODO CHECK BELOW IS OK.
        #self.sqrt_one_minus_alphas_cumprod[self.dirac_z0 == 0] = 0.
        #self.posterior_mean_coef1[self.posterior_mean_coef1/ self.posterior_mean_coef1 != self.posterior_mean_coef1/self.posterior_mean_coef1] = 0.
        #self.posterior_mean_coef2[self.posterior_mean_coef2/ self.posterior_mean_coef2 != self.posterior_mean_coef2/self.posterior_mean_coef2] = 0.

    def _diffuse(self, z_t0, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        assert z_t0.shape[0] == t.shape[0]
        t = t.view(-1)
        noise = torch.randn_like(z_t0)
        mu = self.sqrt_alphas_cumprod[t, :] * z_t0
        sigma = self.sqrt_one_minus_alphas_cumprod[t, :]
        return mu + noise * sigma, (mu, sigma)

    def diffuse(self, z_t0, t, t0=None):
        if t0 is None:
            return self._diffuse(z_t0, t)
        """
        Diffuse the data from t0 to t_end.

        In other words, sample from q(x_t | x_0).
        """
        assert z_t0.shape[0] == t.shape[0] == t0.shape[0]

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t.view(-1), :]/self.sqrt_alphas_cumprod[t0.view(-1), :]
        sqrt_one_minus_alphas_cumprod = (1 - sqrt_alphas_cumprod**2).sqrt()

        mu = sqrt_alphas_cumprod * z_t0
        sigma = sqrt_one_minus_alphas_cumprod
        noise = torch.randn_like(z_t0)

        return mu + noise * sigma, (mu, sigma)

    def _prev_mean_var(self, z_t, z_0, t):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)

        """
        t = t.view(-1)

        mu_cond = self.posterior_mean_coef1[t, :] * z_0 + self.posterior_mean_coef2[t, :] * z_t

        mu_cond = mu_cond

        sigma = self.posterior_variance[t, :]
        return mu_cond, sigma

    def prev_mean_var(self, z_t, z_0, t, t0=None):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_t0)

        """

        if t0 is None:
            return self._prev_mean_var(z_t, z_0, t)

        t = t.view(-1)

        dirac_z0 = self.dirac_z0[t, :]
        #dirac_zt_1 = self.dirac_zt_1[t, :]
        #no_dirac = self.no_dirac[t, :]

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t.view(-1), :] / self.sqrt_alphas_cumprod[t0.view(-1), :]
        sqrt_alphas_cumprod_prev = self.sqrt_alphas_cumprod[t.view(-1)-1, :] / self.sqrt_alphas_cumprod[t0.view(-1), :]
        sqrt_one_minus_alphas_cumprod = (1 - sqrt_alphas_cumprod ** 2).sqrt()
        sqrt_one_minus_alphas_cumprod_prev = (1 - sqrt_alphas_cumprod_prev ** 2).sqrt()
        betas = self.betas[t.view(-1), :]
        alphas = self.alphas[t.view(-1), :]
        posterior_mean_coef1 = betas * sqrt_alphas_cumprod_prev/(sqrt_one_minus_alphas_cumprod)**2
        posterior_mean_coef2 = sqrt_one_minus_alphas_cumprod_prev**2 * alphas.sqrt()/(sqrt_one_minus_alphas_cumprod)**2

        posterior_mean_coef1[dirac_z0 == 1.] = 1.
        posterior_mean_coef2[dirac_z0 == 1.] = 0.

        t = t.view(-1)

        mu_cond = self.posterior_mean_coef1[t, :] * z_0 + self.posterior_mean_coef2[t, :] * z_t
        #mu_cond[mu_cond / mu_cond != mu_cond / mu_cond] = 0.

        #mu_cond = z_t * dirac_zt_1 + dirac_z0 * z_0 + no_dirac * mu_cond
        sigma = sqrt_one_minus_alphas_cumprod_prev**2/sqrt_one_minus_alphas_cumprod**2 * betas
        # TODO define a value for sigma when it is a dirac
        return mu_cond, sigma

    def past_sample(self, mu_z_pred, t_1, temperature=1.):
        t_1 = t_1.view(-1)

        log_sigma = self.posterior_log_variance_clipped[t_1, :]
        log_sigma[log_sigma / log_sigma != log_sigma / log_sigma] = 0.

        no_dirac = self.no_dirac[t_1, :]
        sigma = 0 * (1 - no_dirac) + no_dirac * torch.exp(.5 * log_sigma)
        noise = torch.randn_like(mu_z_pred)

        z_pred = (mu_z_pred + noise * sigma) * (1 - self.dirac_zt_1[t_1, :]) + noise * self.dirac_zt_1[t_1, :]

        return z_pred

    '''
    def reverse(self, z_t0, z_t, t_1):
        t_1 = t_1.view(-1)

        dirac_z0 = self.dirac_z0[t_1, :]
        dirac_zt_1 = self.dirac_zt_1[t_1, :]
        no_dirac = self.no_dirac[t_1, :]
        alphas = self.alphas
        betas = self.betas
        alphas_t = self.alphas_t

        mult_z0 = alphas[t_1, :].sqrt() * betas[t_1 + 1, :] / (1 - alphas[t_1 + 1, :])
        mult_zt = alphas_t[t_1 + 1, :].sqrt() * (1 - alphas[t_1, :]) / (1 - alphas[t_1 + 1, :])
        mult_z0[mult_z0 / mult_z0 != mult_z0 / mult_z0] = 0.
        mult_zt[mult_zt / mult_zt != mult_zt / mult_zt] = 1.

        mu_cond = mult_z0 * z_t0 + mult_zt * z_t

        sigma_cond = ((1 - self.alphas[t_1, :]) / (1 - self.alphas[t_1 + 1, :]) * self.betas[t_1 + 1, :]).sqrt()

        mu_cond[mu_cond/mu_cond != mu_cond/mu_cond] = 0.
        sigma_cond[sigma_cond / sigma_cond != sigma_cond / sigma_cond] = 0.

        mu = z_t * dirac_zt_1 + dirac_z0 * z_t0 + no_dirac * mu_cond

        return mu #+ sigma_cond * torch.randn_like(z_t)

    def past_sample(self, mu_z_pred, t_1, temperature=1.):
        sigma_cond = ((1 - self.alphas[t_1.view(-1), :]) / (1 - self.alphas[t_1.view(-1) + 1, :]) * self.betas[t_1.view(-1) + 1, :]).sqrt()
        sigma_cond[sigma_cond / sigma_cond != sigma_cond / sigma_cond] = 0.
        no_dirac = self.no_dirac[t_1.view(-1), :]
        sigma_cond = 0 * (1 - no_dirac) + no_dirac * sigma_cond
        return mu_z_pred + torch.randn_like(mu_z_pred) * sigma_cond * temperature#self.betas[t_1.view(-1)+1, :].sqrt()
    '''


class TransitionNet(nn.Module):
    def __init__(self, z_dim, layers, t_dim=1, diffuser=None, pos_enc=None, act=nn.SELU, simplified_trans=False):
        super(TransitionNet, self).__init__()
        layers = [z_dim + t_dim] + layers + [z_dim]
        net = []
        for l1, l2 in zip(layers[0][:-1], layers[0][1:]):
            net += [nn.Linear(l1, l2), act()]
        net.pop()
        self.net = nn.Sequential(*net)
        self.diffuser = diffuser
        self.z_dim = z_dim
        self.device = 'cpu'
        self.pos_enc = pos_enc
        self.simplified_trans = simplified_trans

    def forward(self, z, t):

        t = self.pos_enc(t) if self.pos_enc is not None else t

        mu_z_pred = self.net(torch.cat((z, t), 1)) + z
        return mu_z_pred#self.net(torch.cat((z, t), 1)) #+ z


    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T - 1, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t, t_t)
            if self.simplified_trans:
                z_t = None
            else:
                z_t = self.diffuser.past_sample(mu_z_pred, t_t, temperature)
        #print(z_t.norm(), z_t.std(), z_t.mean())
        return z_t



class ImprovedTransitionNet(nn.Module):
    def __init__(self, z_dim, layers, t_dim=1, diffuser=None, pos_enc=None, act=nn.SELU, simplified_trans=False, device='cpu'):
        super(ImprovedTransitionNet, self).__init__()
        self.nets = nn.ModuleList()
        for l in layers:
            net = []
            layer = [z_dim + t_dim] + l + [z_dim]
            for l1, l2 in zip(layer[:-1], layer[1:]):
                net += [nn.Linear(l1, l2), act()]
            net.pop()
            self.nets.append(nn.Sequential(*net))
        #net.pop()
        #self.net = nn.Sequential(*net)
        self.diffuser = diffuser
        self.z_dim = z_dim
        self.device = device
        self.pos_enc = pos_enc
        self.simplified_trans = simplified_trans

    def forward(self, z, t):

        t_enc = self.pos_enc(t) if self.pos_enc is not None else t
        t = t.view(-1)
        out = z

        if self.simplified_trans:
            for net in self.nets:
                out = net((torch.cat((z, t_enc), 1)))

            denom = (1 - self.diffuser.alphas_cumprod[t, :]).sqrt()
            denom[denom == 0.] = 1e-6
            factor = self.diffuser.betas[t, :]/denom
            div = self.diffuser.alphas[t, :].sqrt()
            div[div == 0.] = 1e-6

            deterministic = (self.diffuser.betas[t, :] == 0.).float()
            out = (1-deterministic) * (z - factor * out)/div + deterministic * z
        else:
            for net in self.nets:
                out = net((torch.cat((z, t_enc), 1))) + out
        return out


    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t, t_t)

            z_t = self.diffuser.past_sample(mu_z_pred, t_t, temperature)
        #print(z_t.norm(), z_t.std(), z_t.mean())
        return z_t


# Only ok for 16*16 images
class UNetTransitionNet(nn.Module):
    def __init__(self, z_dim, t_dim=1, diffuser=None, pos_enc=None, act=nn.SELU, simplified_trans=False, device='cpu',
                 cond_in=24):
        super(UNetTransitionNet, self).__init__()
        if z_dim[1] == z_dim[2] == 16:
            init_channels = z_dim[0]
            kernel_size = 4
            self.act = act()
            self.conv1 = nn.Conv2d(init_channels, init_channels * 4, kernel_size, padding=1, stride=2)
            self.conv2 = nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size, padding=1, stride=2)
            self.conv3 = nn.Conv2d(init_channels * 8, init_channels * 8, kernel_size, padding=0, stride=2)

            self.t_conv1 = nn.ConvTranspose2d(init_channels * 8 + cond_in, init_channels * 4, 4, 1, 0, bias=False)
            self.t_conv2 = nn.ConvTranspose2d(init_channels * 8 + init_channels * 4, init_channels*4, 4, 2, 1, bias=False)
            self.t_conv3 = nn.ConvTranspose2d(init_channels * 8, init_channels * 2, 4, 2, 1, bias=False)

            self.conv4 = nn.Conv2d(init_channels * 2, init_channels * 4, 3, padding=1, stride=1)
            self.conv5 = nn.Conv2d(init_channels * 4, init_channels * 1, 3, padding=1, stride=1)

            self.conv1_attention = nn.Sequential(nn.Linear(t_dim, init_channels * 4), nn.Sigmoid())
            self.conv2_attention = nn.Sequential(nn.Linear(t_dim, init_channels * 8), nn.Sigmoid())
            self.t_conv1_attention = nn.Sequential(nn.Linear(t_dim, init_channels * 4), nn.Sigmoid())
            self.t_conv2_attention = nn.Sequential(nn.Linear(t_dim, init_channels * 4), nn.Sigmoid())
            self.t_conv3_attention = nn.Sequential(nn.Linear(t_dim, init_channels * 2), nn.Sigmoid())
        else:
            raise Exception

        self.diffuser = diffuser
        self.z_dim = z_dim
        self.z_dim_tot = z_dim[0] * z_dim[1] * z_dim[2]
        self.device = device
        self.pos_enc = pos_enc
        self.simplified_trans = simplified_trans

    def _forward(self, z, t, cond):
        t = self.pos_enc(t) if self.pos_enc is not None else t

        h1 = self.conv1(z)
        h1_timed = self.conv1_attention(t).unsqueeze(2).unsqueeze(3) * h1
        h2 = self.conv2(self.act(h1))
        h2_timed = self.conv2_attention(t).unsqueeze(2).unsqueeze(3) * h2
        h3 = self.conv3(self.act(h2))
        h3 = self.t_conv1_attention(t).unsqueeze(2).unsqueeze(3) * self.act(self.t_conv1(torch.cat((cond, h3), 1)))

        h4 = self.t_conv2_attention(t).unsqueeze(2).unsqueeze(3) * self.act(self.t_conv2(torch.cat((h3, h2_timed), 1)))

        h5 = self.t_conv3_attention(t).unsqueeze(2).unsqueeze(3) * self.act(self.t_conv3(torch.cat((h4, h1_timed), 1)))

        out = self.conv5(self.act(self.conv4(h5)))

        return out

    def forward(self, z, t, cond):
        dims = z.shape
        b_size = dims[0]
        out = self._forward(z, t, cond)
        t = t.view(-1)

        if self.simplified_trans:

            denom = (1 - self.diffuser.alphas_cumprod[t, :]).sqrt()
            denom[denom == 0.] = 1e-6
            factor = self.diffuser.betas[t, :]/denom
            div = self.diffuser.alphas[t, :].sqrt()
            div[div == 0.] = 1e-6

            deterministic = (self.diffuser.betas[t, :] == 0.).float()
            out = (1-deterministic) * (z.view(b_size, -1) - factor * out.view(b_size, -1))/div + deterministic * z.view(b_size, -1)
            out = out.view(*dims)

        return out

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, cond, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim_tot).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t.view(nb_samples, *self.z_dim), t_t, cond).view(nb_samples, -1)

            z_t = self.diffuser.past_sample(mu_z_pred, t_t, temperature)

        return z_t.view(nb_samples, *self.z_dim)


class DDPMUNetTransitionNet(nn.Module):
    def __init__(self, z_dim, diffuser=None, simplified_trans=False, device='cpu'):
        super(DDPMUNetTransitionNet, self).__init__()
        self.net = create_model(64, 128, 2, False, False, False, '16, 8', 4, -1, True, 0., False, 24,
                                in_channels=z_dim[0], out_channels=z_dim[0])
        self.diffuser = diffuser
        self.z_dim = z_dim
        self.z_dim_tot = z_dim[0] * z_dim[1] * z_dim[2]
        self.device = device
        self.simplified_trans = simplified_trans

    def _forward(self, z, t, cond):
        out = self.net(z, t.view(-1), cond=cond.squeeze(2).squeeze(2))
        return out

    def forward(self, z, t, cond):
        dims = z.shape
        b_size = dims[0]
        out = self._forward(z, t, cond)
        t = t.view(-1)

        if self.simplified_trans:

            denom = (1 - self.diffuser.alphas_cumprod[t, :]).sqrt()
            denom[denom == 0.] = 1e-6
            factor = self.diffuser.betas[t, :]/denom
            div = self.diffuser.alphas[t, :].sqrt()
            div[div == 0.] = 1e-6

            deterministic = (self.diffuser.betas[t, :] == 0.).float()
            out = (1-deterministic) * (z.view(b_size, -1) - factor * out.view(b_size, -1))/div + deterministic * z.view(b_size, -1)
            out = out.view(*dims)

        return out

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, cond, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim_tot).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t.view(nb_samples, *self.z_dim), t_t, cond).view(nb_samples, -1)

            z_t = self.diffuser.past_sample(mu_z_pred, t_t, temperature)

        return z_t.view(nb_samples, *self.z_dim)





'''
class NFTransitionNet(nn.Module):
    def __init__(self, z_dim, layers, t_dim=1, diffuser=None, pos_enc=None, act=nn.SELU, simplified_trans=False):
        super(NFTransitionNet, self).__init__()
        cond_type = AutoregressiveConditioner
        conf_args = {'in_size': z_dim, "hidden": layers, "out_size": 2, 'cond_in': t_dim}
        norm_type = AffineNormalizer
        norm_args = {}
        nb_flow = 3
        self.net = buildFCNormalizingFlow(nb_flow, cond_type, conf_args, norm_type, norm_args)
        self.diffuser = diffuser
        self.z_dim = z_dim
        self.device = 'cpu'
        self.pos_enc = pos_enc
        self.simplified_trans = simplified_trans

    def forward(self, z, t):

        t = self.pos_enc(t) if self.pos_enc is not None else t

        mu_z_pred, jac = self.net(z, t) #+ z
        return mu_z_pred#self.net(torch.cat((z, t), 1)) #+ z


    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples, t0=0, temperature=1.):
        if self.diffuser is None:
            raise NotImplementedError

        zT = torch.randn(nb_samples, self.z_dim).to(self.device) * temperature
        T = self.diffuser.T
        z_t = zT
        for t in range(T - 1, t0-1, -1):
            t_t = torch.ones(nb_samples, 1).to(self.device).long() * t

            mu_z_pred = self.forward(z_t, t_t)
            if self.simplified_trans:
                alpha_bar_t = self.diffuser.alphas[t_t.view(-1) + 1, :]
                alpha_t = self.diffuser.alphas_t[t_t.view(-1) + 1, :]
                beta_t = self.diffuser.betas[t_t.view(-1) + 1, :]

                is_dirac = torch.logical_or((1 - alpha_bar_t) == 0., alpha_t == 1.)
                beta_t[is_dirac] = 1.
                alpha_bar_t[is_dirac] = 0.
                alpha_t[is_dirac] = 1.
                #print(alpha_t.sqrt().min())
                #print(((1 - alpha_t)/(1 - alpha_bar_t).sqrt()).max())
                mu_z_pred = (z_t - beta_t/(1 - alpha_bar_t).sqrt() * mu_z_pred)/alpha_t.sqrt()
                z_t = z_t * is_dirac.float() + (1 - is_dirac.float()) * self.diffuser.past_sample(mu_z_pred, t_t + 1)
                #print(z_t.norm(), z_t.std(), z_t.mean())
            else:
                z_t = self.diffuser.past_sample(mu_z_pred, t_t + 1)
        #print(z_t.norm(), z_t.std(), z_t.mean())
        return z_t




m = AsynchronousDiffuser([0.0001], [.02], [0], [1000], [1])

z0 = torch.randn(10, 1)
t = torch.randint(1, 1000, (10, 1))
zt, _ = m.diffuse(z0, t)
print(zt)
print(m.prev_mean_var(zt, z0, t)[1], m.prev_mean_var(zt, z0, t, t*0)[1])
'''
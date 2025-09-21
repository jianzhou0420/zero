import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, depth=5):
        super(Encoder, self).__init__()
        self.fc_layers = nn.ModuleList()
        for _ in range(depth):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.LeakyReLU(layer(x))
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)                     # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, depth=5):
        super(Decoder, self).__init__()
        self.fc_layers = nn.ModuleList()
        for _ in range(depth):
            self.fc_layers.append(nn.Linear(latent_dim, hidden_dim))
            latent_dim = hidden_dim

        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.LeakyReLU(layer(x))
        x_hat = torch.sigmoid(self.FC_output(x))
        return x_hat


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        input_dim = config['VAE']['input_dim']
        output_dim = config['VAE']['output_dim']
        hidden_dim = config['VAE']['hidden_dim']
        latent_dim = config['VAE']['latent_dim']
        encoder_depth = config['VAE']['encoder_depth']
        decoder_depth = config['VAE']['decoder_depth']

        self.Encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=encoder_depth)
        self.Decoder = Decoder(output_dim=output_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=decoder_depth)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')        # sampling epsilon
        z = mean + var * epsilon                          # reparameterization trick
        return z

    def forward(self, batch):
        input = batch['input']
        output = batch['output']
        mean, log_var = self.Encoder(input)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        loss = self.loss_function(output, x_hat, mean, log_var)  # calculate loss
        return loss

    @staticmethod
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = F.mse_loss(x_hat, x, reduction='mean')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return {
            'recon_loss': reproduction_loss,
            'KLD': KLD,
            'total_loss': reproduction_loss + KLD
        }

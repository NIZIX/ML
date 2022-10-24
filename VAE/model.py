import torch
import torch.nn as nn
from torchsummary import summary


class Reshape(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def forward(self, x):
        return x.reshape(x.shape[0], self.params[0], self.params[1], self.params[2])


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # <определите архитектуры encoder и decoder
        # помните, у encoder должны быть два "хвоста", 
        # т.е. encoder должен кодировать картинку в 2 переменные -- mu и logsigma>
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=(3,3), stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(6400, 1200)
        )

        self.decoder = nn.Sequential(
            nn.Linear(600, 6*6*32),
            Reshape([32, 6, 6]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, (4,4), stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, (4,4), stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, (6,6), stride=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        # <реализуйте forward проход энкодера
        # в качестве ваозвращаемых переменных -- mu и logsigma>
        x = self.encoder(x).view(-1, 2, 600)
        mu = x[:, 0, :]
        logsigma = x[:, 1, :]
        
        return mu, logsigma
    
    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            # <засемплируйте латентный вектор из нормального распределения с параметрами mu и sigma>
            std = torch.exp(0.5 * logsigma)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
            return sample
        else:
            # на инференсе возвращаем не случайный вектор из нормального распределения, а центральный -- mu. 
            # на инференсе выход автоэнкодера должен быть детерминирован.
            return mu
    
    def decode(self, z):
        reconstruction = self.decoder(z)
        
        return reconstruction

    def forward(self, x):
        # <используя encode и decode, реализуйте forward проход автоэнкодера
        # в качестве ваозвращаемых переменных -- mu, logsigma и reconstruction>
        x = self.encoder(x).view(-1, 2, 600)
        mu = x[:, 0, :]
        logsigma = x[:, 1, :]

        repres = self.gaussian_sampler(mu, logsigma)

        return self.decoder(repres), mu, logsigma

def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений разных людей
    """
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss

def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции (как mse в обычном autoencoder)
    """
    loss = nn.BCELoss(reduction="sum")
    return loss(reconstruction, x)

def loss_vae(x, mu, logsigma, reconstruction):
    # print(KL_divergence(mu, logsigma))
    # print(log_likelihood(x, reconstruction))
    return KL_divergence(mu, logsigma) + log_likelihood(x, reconstruction)


model = VAE().to("cuda:0")
summary(model, (3, 64, 64))
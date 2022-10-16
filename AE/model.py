import torch
import torch.nn as nn
from torchsummary import summary

class Resize(nn.Module):
    def __init__(self, dim) -> None:
        super(Resize, self).__init__()
        self.dim = dim

        if (len(dim) != 3):
            raise Exception("Expected list or tuple size of 3")

    def forward(self, x):
        return x.view(x.shape[0], self.dim[0], self.dim[1], self.dim[2])


class AE(nn.Module):
    def __init__(self, latent_size) -> None:
        super(AE, self).__init__()

            # 128 * 128 * 3
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=2, padding=2), # [-1, 32, 64, 64]
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=2, padding=0), # [-1, 64, 29, 29]
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=2, padding=0), # [-1, 128, 12, 12]
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, latent_size), 
            nn.LeakyReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128 * 12 * 12),
            Resize([128, 12, 12]),
            nn.ConvTranspose2d(128, 64, kernel_size=(7, 7), stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=2, padding=0, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent_code = self.encoder(x)

        reconstruction = self.decoder(latent_code)

        return reconstruction, latent_code


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AE(256).to(device)
print(summary(model, (3, 128, 128)))
# src/model.py

import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        # 2D-Conv Encoder (shape : [Batches, channels, image(h), image(w)])
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='zeros'
                ), # shape : (B, 1, 28, 28) -> (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='zeros'
                ), # shape : (B, 16, 14, 14) -> (B, 64, 7, 7)
            nn.ReLU()
        )

        # avg of Latent vecs (shape : [Batches, latent_dim])
        self.fc_mu = nn.Linear(
                        in_features=64*7*7, 
                        out_features=latent_dim
                        )

        # ln(variance) of Latent vecs (shape: [Batches, latent_dim])
        self.fc_logvar = nn.Linear(
                            in_features=64*7*7, 
                            out_features=latent_dim
                            )

        # 2D-Conv Decoder (shape : [Batches, ...SAME SHAPE WITH ORIGINAL DATA...])
        self.fc_decoder = nn.Linear(
                                in_features=latent_dim, 
                                out_features=64*7*7
                                )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, 
                out_channels=16, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16, 
                out_channels=1, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                padding_mode='zeros'
            ),
            nn.Sigmoid()
        )

        # Params Init(Xavier uniform initialization)
        self._init_weights()

    
    def encode(self, x):
        encoded_x = self.encoder(x)
        encoded_x_flatten = encoded_x.reshape(encoded_x.size(0), -1) # shape : (Batches, flatten_len of sample)
        return self.fc_mu(encoded_x_flatten), self.fc_logvar(encoded_x_flatten)


    def reparameterize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar) # standard deviation of Latent vecs
        eps = torch.randn_like(std_dev)
        
        return mu + eps * std_dev
    

    def decode(self, latent_z):
        preprocessed_z_in_2d = self.fc_decoder(latent_z).view(-1, 64, 7, 7) # shape : (Batches, 64, 7, 7)
       
        return self.decoder(preprocessed_z_in_2d)
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_z = self.reparameterize(mu, logvar)
        reconst_x = self.decode(latent_z)

        return reconst_x, mu, logvar, latent_z
    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class FlattenVAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        # Linear Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # avg of Latent vecs
        self.fc_mu = nn.Linear(128, latent_dim)

        # ln(variance) of Latent vecs
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Linear Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

        # Params Init(Xavier uniform initialization)
        self._init_weights()


    def encode(self, x):
        encoded_x = self.encoder(x)

        return self.fc_mu(encoded_x), self.fc_logvar(encoded_x)


    def reparameterize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar) # standard deviation of Latent vecs
        eps = torch.randn_like(std_dev)
        
        return mu + eps * std_dev
    

    def decode(self, latent_z):
        decoded_x = self.decoder(latent_z)
 
        return decoded_x.view(-1, 1, 28, 28)


    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_z = self.reparameterize(mu, logvar)
        reconst_x = self.decode(latent_z)

        return reconst_x, mu, logvar, latent_z
    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class ConvAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        # 2D-Conv Encoder (shape : [Batches, channels, image(h), image(w)])
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='zeros'
                ), # shape : (B, 1, 28, 28) -> (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='zeros'
                ), # shape : (B, 16, 14, 14) -> (B, 64, 7, 7)
            nn.ReLU()
        )

        # Latent vecs (shape : [Batches, latent_dim])
        self.fc_z = nn.Linear(
                        in_features=64*7*7, 
                        out_features=latent_dim
                        )

        # 2D-Conv Decoder (shape : [Batches, ...SAME SHAPE WITH ORIGINAL DATA...])
        self.fc_decoder = nn.Linear(
                                in_features=latent_dim, 
                                out_features=64*7*7
                                )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, 
                out_channels=16, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16, 
                out_channels=1, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                padding_mode='zeros'
            ),
            nn.Sigmoid()
        )

        # Params Init(Xavier uniform initialization)
        self._init_weights()   


    def encode(self, x):
        encoded_x = self.encoder(x)
        encoded_x_flatten = encoded_x.reshape(encoded_x.size(0), -1) # shape : (Batches, flatten_len of sample)

        return self.fc_z(encoded_x_flatten)
    

    def decode(self, latent_z):
        preprocessed_z_in_2d = self.fc_decoder(latent_z).view(-1, 64, 7, 7) # shape : (Batches, 64, 7, 7)
        
        return self.decoder(preprocessed_z_in_2d)


    def forward(self, x):
        latent_z = self.encode(x)
        reconst_x = self.decode(latent_z)

        return reconst_x, None, None, latent_z
    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class FlattenAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        # Linear Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Linear Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

        # Params Init(Xavier uniform initialization)
        self._init_weights()

    
    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, latent_z):
        decoded_x = self.decoder(latent_z)
        return decoded_x.view(-1, 1, 28, 28) 
    

    def forward(self, x):
        latent_z = self.encode(x)
        reconst_x = self.decode(latent_z) 

        return reconst_x, None, None, latent_z
    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

def reparametrize(mu,logvar, training=True):
    if(not training):
       return mu
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAE_Simple(nn.Module):
    def __init__(self, feature_dim=2048, latent_dim=128):
        super(VAE_Simple, self).__init__()
        self.fc = nn.Linear(feature_dim, 128)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.activation(self.fc(x))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar, reparametrize(mu, logvar, self.training) # just handle these here!


class NearestNeighborContrastiveI3D(nn.Module):
    def __init__(self, feature_dim=2048, projection_dim=128):
        super(NearestNeighborContrastiveI3D, self).__init__()

        # Intra-video projection head
        self.intra_projector = VAE_Simple(feature_dim, projection_dim)

        # Inter-video projection head
        self.inter_projector = VAE_Simple(feature_dim, projection_dim)

        # Now make a decoder for the intra and inter projections
        self.intra_decoder = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.inter_decoder = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        # Pass features through projection heads
        mu_intra, logvar_intra, intra_embeddings = self.intra_projector(features)
        mu_inter, logvar_inter, inter_embeddings = self.inter_projector(features)

        decoded_intra = self.intra_decoder(intra_embeddings)
        decoded_inter = self.inter_decoder(inter_embeddings)

        return intra_embeddings, inter_embeddings, decoded_inter, decoded_intra, [mu_intra, logvar_intra], [mu_inter, logvar_inter]

    def from_latent_space(self, latent_features, both=False):
        # Pass features through projection heads
        # Given latent features, decode them to get the original features and original feature size.
        if both:
            decoded_intra = self.intra_decoder(latent_features[0])
            decoded_inter = self.inter_decoder(latent_features[1])
        else:
            decoded_intra = self.intra_decoder(latent_features)
            decoded_inter = decoded_intra
        return decoded_inter, decoded_intra
  

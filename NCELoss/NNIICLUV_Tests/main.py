import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

from custom_queue import Queue
from model import NearestNeighborContrastiveI3D
from loss import InfoNCELoss

def load_features(path):
    # Load features from path
    return torch.load(path)


def nn_magic(intra_embeddings, temporal, embedding_dim, myQueue, debug=False): # In future get K input
  batch_size, temporal, embedding_dim = intra_embeddings.shape
  intra_embeddings = intra_embeddings.reshape(batch_size * temporal, embedding_dim)
  nn_indices, nn_embeddings, nn_labels = myQueue.find_nearest_neighbors(intra_embeddings)
  nn_embeddings = nn_embeddings.reshape(batch_size, temporal, embedding_dim)
  nn_labels = nn_labels.reshape(batch_size, temporal, 3)
  nn_indices = nn_indices.reshape(batch_size, temporal)
  return nn_indices, nn_embeddings, nn_labels


def getRandomFeaturesNLabels(batch_size, temporal, feature_dim, num_classes, device='cuda', epoch=0):
  features = torch.randn(batch_size, temporal, feature_dim)
  labels = torch.zeros(batch_size, 3)
  for i in range(batch_size):
    labels[i] = torch.tensor([i,random.randint(0,num_classes),epoch]) # Epoch 0
  return features.to(device), labels.to(device)

# Example usage
if __name__ == "__main__":
    
    #features = load_features("features.pth")
    # Dummy feature input
    batch_size = 2
    temporal = 50
    feature_dim = 2048
    embedding_dim = 128
    num_classes = 20
    queue_size = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Form Model
    myModel = NearestNeighborContrastiveI3D(feature_dim=feature_dim, projection_dim=embedding_dim).to(device)
    # Form Queue
    myQueue = Queue(queue_size=queue_size, embedding_dim=embedding_dim, device=device)
    # Form Loss
    loss = InfoNCELoss()
    # Form Optimizer
    optimizer = torch.optim.Adam(myModel.parameters(), lr=1e-3)
    
    
    # Form Initial Features and labels
    features ,labels = getRandomFeaturesNLabels(batch_size, temporal, feature_dim, num_classes)
    # Store initial features
    intra_embeddings, inter_embeddings = myModel(features)
    myQueue.enqueue(intra_embeddings, labels)
    
    
    # The Main Loop
    # Reform after initial features
    num_epochs = 100

    for epoch in range(num_epochs):
        features ,labels = getRandomFeaturesNLabels(batch_size, temporal, feature_dim, num_classes, epoch=epoch)
        intra_embeddings, inter_embeddings = myModel(features)
        nn_indices, positives, nn_labels = nn_magic(intra_embeddings, temporal, embedding_dim, myQueue)
        negative_embeddings, negative_indexes = myQueue.getNegatives(nn_indices)
        loss_value = loss(intra_embeddings, positives, negative_embeddings)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        print(f'Epoch: {epoch} Loss: {loss_value.item()}')
        with torch.no_grad(): # Update the queue with updated features
            intra_embeddings, inter_embeddings = myModel(features)
            myQueue.enqueue(intra_embeddings, labels)
    exit()

    model = NearestNeighborContrastiveI3D(feature_dim=2048, projection_dim=128)

    # Dummy input: Batch of video clips (batch_size=8, channels=3, frames=16, height=112, width=112)

    # Forward pass
    intra_embeddings, inter_embeddings = model(features)
    print(f"Intra-video embeddings shape: {intra_embeddings.shape}")
    print(f"Inter-video embeddings shape: {inter_embeddings.shape}")

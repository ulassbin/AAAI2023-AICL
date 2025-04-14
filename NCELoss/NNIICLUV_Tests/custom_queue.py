import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from NCELoss.NNIICLUV_Tests.vid_fft import torch_fft

class Queue():
    def __init__(self, queue_size=65536, embedding_dim=128, device='cuda'):
        self.device = device
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.queue = torch.zeros((queue_size, embedding_dim)).to(device)  # Initialize with zeros
        self.label_queue = torch.zeros((queue_size, 3), dtype=int).to(device) # vid_id, class_id, epoch_id
        self.vid_queue = []
        self.ptr = 0  # Pointer to track enqueueing position
        self.overflown = False

    def subsample_negative_indexes(self, negative_indexes, new_size):
        batch_size, temporal, original_size = negative_indexes.shape
        if(new_size >= original_size):
            return negative_indexes
        random_indices = torch.randint(0, original_size, (batch_size, temporal, new_size), dtype=torch.long, device=negative_indexes.device)
        return torch.gather(negative_indexes, 2, random_indices)

    def getNegatives(self, nn_indices, sample=True):
        # Given indexes of size batch, temporal or just temporal get the negatives
        if(nn_indices.dim() == 1):
            nn_indices = nn_indices.unsqueeze(0)
        batch_size, temporal = nn_indices.shape
        actual_queue_size = self.ptr+1 if not self.overflown else self.queue_size
        index_list = [list(range(actual_queue_size)) for i in range(batch_size)]
        sample_size = 100
        min_size = min(sample_size, actual_queue_size)
        negative_embeddings = torch.zeros(batch_size, temporal, min_size, self.embedding_dim)
        negative_indexes = torch.zeros(batch_size, temporal, min_size, dtype=int)
        for i in range(batch_size):
          for j in range(temporal):
            if(sample):
              q_ids = self.sample_queue_without_indices(sample_size, nn_indices[i][j])
            else:
              q_ids = self.get_queue_without_indices(nn_indices[i][j])
            negative_indexes[i][j] = q_ids
            negative_embeddings[i][j] = self.queue[q_ids] # Get the embeddings
          return negative_embeddings.to(self.device), negative_indexes.to(self.device)

    def append(self, embeddings, labels=None):
      batch_size, t, feature_dim = embeddings.shape # Assuming fixed t
      num_snips = batch_size*t
      if labels is not None:
        labels = labels.reshape(-1,3)
        self.label_queue[self.ptr:self.ptr + num_snips] = copy.deepcopy(labels)

      self.queue[self.ptr:self.ptr + num_snips] = copy.deepcopy(embeddings.reshape(-1,feature_dim).detach())
      for i in range(batch_size):
        indexes = torch.arange(self.ptr + i*t, self.ptr + i*t + t)
        self.vid_queue.append(torch.tensor(indexes, dtype=int))
      
      if(not self.overflown and self.ptr + num_snips >= self.queue_size):
        self.overflown = True
      self.ptr = (self.ptr + num_snips) % self.queue_size
    
    def shift(self, amount):
        self.overflown = True
        self.queue = torch.roll(self.queue, -amount, 0)
        self.label_queue = torch.roll(self.label_queue, -amount, 0)
        del_indexes = []

        # Correct video queue
        for i in range(len(self.vid_queue)):
            new_indexes = self.vid_queue[i] - amount
            # now count valid
            valid = new_indexes[new_indexes >= 0]
            if len(valid) == 0:
              del_indexes.append(i)
            else:
              self.vid_queue[i] = valid
        for i in sorted(del_indexes, reverse=True):
            del self.vid_queue[i]
        self.ptr -= amount

    def reshape_labels(self, labels, t):
        if labels is None:
            return None
        return labels.reshape(-1,1,3).repeat(1, t, 1)

    def enqueue(self, embeddings, labels=None):
        batch_size, temporal, feature_dim = embeddings.shape
        labels = self.reshape_labels(labels, temporal)
        num_items = batch_size * temporal
        if self.ptr + num_items > self.queue_size:
            overflow = (self.ptr + num_items) - self.queue_size
            self.shift(overflow) # shift the queue
            self.append(embeddings, labels)
        else:
          self.append(embeddings, labels)

    def find_nearest_neighbors(self, query_embeddings):
      query_norm = F.normalize(query_embeddings, dim=1)
      queue_norm = F.normalize(self.queue, dim=1)
      similarities = torch.matmul(query_norm, queue_norm.T)  # Shape: (batch_size, queue_size)
      nn_indices = similarities.argmax(dim=1)
      return nn_indices, self.queue[nn_indices], self.label_queue[nn_indices]
    
    def sample_queue(self, amount):
        indices = torch.randperm(self.queue_size)[:amount]
        return indices, self.queue[indices], self.label_queue[indices]

    def sample_queue_without_indices(self, amount, remove_indices):
        actual_queue_size = self.ptr+1 if not self.overflown else self.queue_size
        mask = torch.ones(actual_queue_size, dtype=bool)
        mask[remove_indices] = False
        indices = torch.tensor(list(range(actual_queue_size)))[mask]
        amount = min(amount, mask.sum().item()) # Filtered actual size vs requested size
        indices = indices[:amount]
        return indices

    def find_nearest_neighbours_subset(self, query_embeddings, subset_size):
        small_queue_indices, small_queue, small_labels = self.sample_queue(subset_size)
        query_norm = F.normalize(query_embeddings, dim=1, eps=1e-8)
        small_queue_norm = F.normalize(small_queue, dim=1, eps=1e-8)
        similarities = torch.matmul(query_norm, small_queue_norm.T)
        nn_indices = similarities.argmax(dim=1)
        return nn_indices, small_queue[nn_indices], small_labels[nn_indices]
    
    def getVidIndices(self, max_samples):
        num_vids = len(self.vid_queue)
        targets = np.array(list(range(num_vids)))
        samples = np.random.choice(num_vids, min(num_vids, max_samples), replace=False)
        return torch.tensor(samples, dtype=int)
    
    def getVidData(self, indices):
        max_length = max([len(self.vid_queue[i]) for i in indices])
        padded_vid_data = []
        for i in indices: # not the most efficient way...
            vid_data = self.queue[self.vid_queue[i]]
            padding = torch.zeros((max_length - vid_data.shape[0], self.embedding_dim), device=self.device)
            padded_vid_data.append(torch.cat((vid_data, padding), dim=0))
        return torch.stack(padded_vid_data, dim=0)

    def find_nearest_vids(self, full_embeddings, max_samples=20):
        # We have vids stored in a list called vid_queue
        num_vids = len(self.vid_queue) # How many unique videos we have
        if(num_vids == 0):
            print('Vid queue is currently empty')
            return None, None
        vid_indices = self.getVidIndices(max_samples)
        queued_vid_targets = self.getVidData(vid_indices)
        #print('Target vids shape: ', queued_vid_targets.shape)
        #print('Full embeddings shape: ', full_embeddings.shape)
        distances = torch_fft.fft_distance_2d_batch(full_embeddings, queued_vid_targets) # This might cause memory issues, might need to partition into smaller chunks later.
        closest = distances.argmin(dim=1).cpu().numpy()
        return queued_vid_targets[closest], vid_indices[closest]
       

    def get_queue_without_indices(self, indices):
        mask = torch.ones(self.queue_size, dtype=bool)
        mask[indices] = False
        return self.queue[mask], self.label_queue[mask]
 
import torch
import os
import random
import numpy as np
import time

def print_memory_usage(message=""):
    """Print GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Function to load videos
def load_videos(data_dir, num_videos=None):
    print_memory_usage('Before loading videos')
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Limit the number of videos if specified
    if num_videos is not None and num_videos < len(video_files):
        video_files = random.sample(video_files, num_videos)
    
    videos = [torch.tensor(np.load(file)) for file in video_files]
    print_memory_usage('After loading videos')
    return videos

# Function to normalize a batch of video sequences
def normalize(video_batch):
    return (video_batch - video_batch.mean(dim=(1,2), keepdim=True)) / video_batch.std(dim=(1,2), keepdim=True)

# Function to pad video sequences to the same length within a batch
def pad_videos(video_batch):
    max_len = max(video.shape[0] for video in video_batch)  # Find the max length in the batch
    padded_videos = []
    
    for video in video_batch:
        padding_size = max_len - video.shape[0]
        padded_video = torch.nn.functional.pad(video, (0, 0, 0, padding_size), mode='constant', value=0)
        padded_videos.append(padded_video)
    
    return torch.stack(padded_videos)

def fft_distance_2d_batch(video_batch1, video_batch2):
    # Normalize the video batches
    video_batch1 = normalize(video_batch1)
    video_batch2 = normalize(video_batch2)

    # Get the size of each batch
    n1, d1 = video_batch1.shape[1], video_batch1.shape[2]
    n2, d2 = video_batch2.shape[1], video_batch2.shape[2]

    # Ensure the feature dimension is the same
    assert d1 == d2, "Feature dimensions must match."

    # Determine padding size (sum of temporal dimensions - 1)
    pad_n = n1 + n2 - 1

    # Apply 2D FFT across the entire video (frames, features)

    fft_video1 = torch.fft.fft2(video_batch1, s=(pad_n, d1))
    fft_video2 = torch.fft.fft2(video_batch2, s=(pad_n, d1))

    fft_video1 = fft_video1.unsqueeze(1)
    fft_video2 = fft_video2.unsqueeze(0)
    #print('fft_video1 shape: ', fft_video1.shape)
    #print('fft_video2 shape: ', fft_video2.shape)
    # Element-wise multiplication in frequency domain and summation over feature dimension
    fft_mult = torch.fft.ifft2(fft_video1 * torch.conj(fft_video2))
    #print('fft_mult shape: ', fft_mult.shape)
    convolution_result = torch.real(fft_mult).sum(dim=-1)  # Summing over feature dimension

    # Peak value in the convolution result for each pair (across temporal dimension)
    peak_values = torch.amax(convolution_result, dim=2)

    # Distance as the inverse of peak value (to ensure similarity yields a small distance)
    distances = 1 / (peak_values + 1e-10)  # Add small value to avoid division by zero
    return distances

# Function to compute the distance matrix for a list of videos with batching
def cdist_fft_2d_batched(videos, batch_size=32):
    num_videos = len(videos)
    distance_matrix = torch.zeros((num_videos, num_videos), device=videos[0].device)
    print_memory_usage('After Forming Matrix')
    for i in range(num_videos):
        # Prepare batches for parallel processing
        for j in range(i + 1, num_videos, batch_size):
            end = min(j + batch_size, num_videos)
            batch_videos_j = pad_videos(videos[j:end])  # Pad videos in the batch
            batch_videos_i = pad_videos([videos[i]] * (end - j))

            # Compute distances for the batch
            distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j)

            # Update the distance matrix
            distance_matrix[i, j:end] = distances
            distance_matrix[j:end, i] = distances  # Symmetric matrix
        #del batch_videos_j, batch_videos_i, distances
        torch.cuda.empty_cache()
        print_memory_usage('{} Iteration'.format(i))
    return distance_matrix

def cuda_fft_distances(data_dir, num_videos, batch=32):
    videos = load_videos(data_dir, num_videos)
    videos = [video.to('cuda') for video in videos]  # Move videos to GPU
    print_memory_usage('Moved video to Gpu')
    distance_matrix = cdist_fft_2d_batched(videos, batch_size=batch)
    print_memory_usage('After Calculating Distance Matrix')
    time.sleep(2)
    torch.cuda.empty_cache()
    print_memory_usage('Emptied cache')
    return distance_matrix.to('cpu').numpy()

if __name__ == '__main__':
    video_batch1 = torch.randn(5, 125, 1024)
    video_batch2 = torch.randn(5, 105, 1024)
    distance = fft_distance_2d_batch(video_batch1, video_batch2)
    print(distance)
    # find the closest video to each video
    closest = distance.argmin(dim=1)
    print(closest)
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2


def get_batched_indices(total_indices, sampling_rate, batch_size, padding=0):
    '''Splits a set of indices into batches with a remainder
    
    Args:
        total_indices: int, the total number of indices in the original list
        sampling_rate: int > 0, the subsampling rate to choose a subset of indices
        batch_size: int > 0, the batch size of the output
        
    Output:
        batched_indices: List[np.array[int]], 
            a list of all but the last array of the same length equal to batch_size
            the last array can be of a different length <= batch_size
    '''
    if padding >= batch_size:
        raise ValueError("padding should be less than batch_size")
    
    output = []
    running_idx = 0
    running_batch = []
    
    while True:
        running_batch.append(running_idx)
        running_idx += sampling_rate
        if len(running_batch) == batch_size:
            output.append(running_batch)
            running_batch = []
            running_idx -= 2 * sampling_rate * padding
        if running_idx >= total_indices:
            if len(running_batch):
                output.append(running_batch)
            break
            
    return output



def get_frames_with_indices_sequentially(video_path, indices):
    '''Return processed video frames at the specified indices
    
    Args:
        video_path: str, path to the video
        indices: array-like, an array of indices to be extracted
        
    Output:
        output_frames: List[np.array], a list of frames
        output_timestamps: List[float], a list of timestamps for the frames
    '''
    cap = cv2.VideoCapture(video_path)
    current_index = indices[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
    
    output_frames = []
    output_timestamps = []
    
    while cap.isOpened():
        success, frame = cap.read()

        # close the container and return the results
        if not success or current_index > max(indices):
            cap.release()
            return output_frames, output_timestamps
        
        # collect the frame and its info
        if current_index in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)
            output_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        current_index += 1
          
    else:
        raise Exception("Failed to open video '%s'!.." % video_path)

        
        
def get_frames_with_indices_jumping(video_path, indices):
    '''Return processed video frames at the specified indices
    
    Args:
        video_path: str, path to the video
        indices: array-like, an array of indices to be extracted
        
    Output:
        output_frames: List[np.array], a list of frames
        output_timestamps: List[float], a list of timestamps for the frames
    '''
    output_frames = []
    output_timestamps = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)
            output_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            
        cap.release()
        return output_frames, output_timestamps
    else:
        raise Exception("Failed to open video '%s'!.." % video_path)



        
class BatchedVideoReader(Dataset):
    
    def __init__(self, video_path, sampling_rate, batch_size, padding=0):
        super().__init__()
        self.video_path = video_path
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.padding = padding
        
        
        # video container
        cap = cv2.VideoCapture(self.video_path)
        self.total_indices = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # indices
        self.indices = get_batched_indices(
            total_indices=self.total_indices,
            sampling_rate=self.sampling_rate,
            batch_size=self.batch_size,
            padding=padding
        )
        self.length = len(self.indices)
        
        
        if self.sampling_rate >= 60:
            self.get_frames_with_indices = get_frames_with_indices_jumping
        else:
            self.get_frames_with_indices = get_frames_with_indices_sequentially
        
        
    def __getitem__(self, idx):
        indices = self.indices[idx]
        frames, timestamps = self.get_frames_with_indices(
            video_path=self.video_path, 
            indices=indices, 
        )

        return frames, np.array(timestamps).astype('float32'), np.array(indices)
    
    
    def __len__(self):
        return self.length
        
        
    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        for k, v in self.__dict__.items():
            if k in ['indices', 'get_frames_with_indices']:
                continue
            if k == 'processor':
                v = str(v).replace('\n', '\n\t')[:-2]
            s += f'    {k}={v}\n'
        s += ')'
        return s
    

def _collate_fn(x):
    '''A custom collate fn for Dataloader'''
    return x[0] 


def get_video_loader(video_path, sampling_rate, batch_size, padding, pin_memory=True):
    '''Return a video loader optimized for CPU/GPU parallelization
    
    Args:
        video_path: str, path to the video
        sampling_rate: int > 0, the subsampling rate to choose a subset of indices
        batch_size: int > 0, the batch size of the output
        padding: int >= 0, padding in terms of frames
        pin_memory: True/False, whether to pin output tensors or not
        
    Output:
        loader: pytorch DataLoader, a data loader which samples frames from the video
            output_frames: torch.Tensor[batch_size, 3, H, W], a tensor of frames
            output_timestamps: torch.Tensor[batch_size], a tensor of timestamps for the frames
    '''
    pin_memory_device = 'cuda' if torch.cuda.is_available() and pin_memory else ''
    
    video = BatchedVideoReader(
        video_path=video_path, 
        sampling_rate=sampling_rate, 
        batch_size=batch_size, 
        padding=padding
    )

    video_loader = DataLoader(
        dataset=video, 
        batch_size=1, 
        shuffle=False, 
        drop_last=False, 
        pin_memory=pin_memory, 
        pin_memory_device=pin_memory_device, 
        num_workers=1, 
        collate_fn=_collate_fn
    )
    
    return video_loader
    
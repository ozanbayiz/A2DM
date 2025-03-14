import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDataset(data.Dataset):
    def __init__(
        self,
        data_path = "data/sns_slahmr_64.npz",
        motion_key = "poses",
    ):

        data = np.load(data_path)
        self.data = data[motion_key]
        self.length = self.data.shape[1]

        print("Total number of motions {}".format(len(self.data)))
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.length, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def DATALoader(
        data_path = "data/sns_slahmr_64.npz",
        motion_key = "poses",
        batch_size = 1,
        num_workers = 1):
    
    trainSet = VQMotionDataset(data_path, motion_key)
    # prob = trainSet.compute_sampling_prob()
    # sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

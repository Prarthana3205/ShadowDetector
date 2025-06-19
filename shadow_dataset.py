from torch.utils.data import Dataset
import numpy as np
import os
import torch

class ShadowDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.data_dir = preprocessed_dir
        self.image_files = sorted(os.listdir(preprocessed_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = np.load(image_path)  # assuming .npy format

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, 0  # no label (unsupervised)


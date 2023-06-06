import torch
import numpy as np
import os
import pandas as pd
from skimage import io, transform

# paired_list = pd.read_csv('../paired_list.csv')
# print(paired_list)


def age2ord_vector(age, age_dim=100):
    age_vector = np.zeros(age_dim, dtype = np.float32)
    for i in range(age_dim - age, age_dim):
        age_vector[i] += 1

    return age_vector



class Dataset(torch.utils.data.Dataset):
    def __init__(self, paired_list_csv_path, data_root):
        self.paired_list = pd.read_csv(paired_list_csv_path)
        self.data_root = data_root

    def __len__(self):
        return len(self.paired_list)

    def __getitem__(self, idx):
        paired_set_info = self.paired_list.loc[idx]
        young_file_path = self.data_root + paired_set_info['young_filename']

        young_age = age2ord_vector(int(paired_set_info['young_age']))
        old_file_path = self.data_root + paired_set_info['old_filename']
        old_age = age2ord_vector(int(paired_set_info['old_age']))

        young_file = np.expand_dims(np.load(young_file_path), axis=0)
        old_file = np.expand_dims(np.load(old_file_path), axis=0)

        return young_file[:,5:213, 11:171], old_file[:,5:213, 11:171], young_age, old_age



# dataset = Dataset('../paired_image_files.csv', "/mnt/d/camcan/np_data/original/")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)
# data = next(iter(dataloader))
# print(data[3].shape)
# print(data[3].dtype)
from utils.data_loader import Dataset
import torch
from configurations import experiment_params
from models.training import Synthesized_model



dataset = Dataset('./paired_image_files.csv', experiment_params.data_root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = experiment_params.batch_size, shuffle = True)


model = Synthesized_model(experiment_params)
model.train(dataloader, experiment_params.epochs)

# for i, data in enumerate(dataloader):
#     print(i, data[0].shape, data[1].shape, data[2].shape, data[3].shape)
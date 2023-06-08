lr = 0.0001
decay = 0.0001
latent_space = 130
batch_size = 64
input_shape = (208, 160, 1)
filters = 32
age_dim = 100
decay = 0.0001
beta_1 = 0.5
beta_2 =0.999
critic_iter = 5
use_cuda = True
gp_weight = 10
id_weight = 100
self_rec_weight = 10
epochs = 600


data_root = "/home/leehu/project/Aged_brain_simulating/data/original/"
paired_list_csv = '/home/leehu/project/Aged_brain_simulating/Aged_brain_simulating/paired_image_files.csv'
model_save_path = "/home/leehu/project/Aged_brain_simulating/Aged_brain_simulating/result/result_1_2023-06-08/model/"
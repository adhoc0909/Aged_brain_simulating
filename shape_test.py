from models.generator import Generator
from models.discriminator import Discriminator
import numpy as np
import easydict
import torch
conf = {'filters':32, 'latent_space':130, 'age_dim':100}
gen_model = Generator(easydict.EasyDict(conf))

age_gap = torch.randn([16, 100])
gen_output = gen_model(torch.randn([16, 1, 208, 160]), age_gap)
print(gen_output.shape)

critic_model = Discriminator(easydict.EasyDict(conf))

critic_output = critic_model(gen_output, age_gap)
print(critic_output.shape)
print(critic_output)

#print(torch.randint(low = 16, high = 100, size = [16, 100]))

#print(gen_model)
#print(critic_model)
from models.generator import Generator
from models.discriminator import Discriminator
import torch
import torch.optim as optim
from torch import autograd
import time as t
import os
from torchvision import utils
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid
from utils.data_loader import age2ord_vector
from utils.cost import identity_loss, self_rec_loss
import numpy as np
import os


def get_age_gap(old_batch, young_batch):
    return torch.from_numpy(np.array(list(map(age2ord_vector, (old_batch - young_batch).sum(axis = 1).type('torch.ByteTensor')))))

class Synthesized_model():
    def __init__(self, conf):
        self.G = Generator(conf)
        self.D = Discriminator(conf)
        self.conf = conf
        self.learning_rate = conf.lr
        self.batch_size = conf.batch_size
        self.b1 = conf.beta_1 # 원본 코드에서 0으로 설정되어 있는데 확인해볼 것
        self.b2 = conf.beta_2
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay = conf.decay)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay = conf.decay)

        self.critic_iter = conf.critic_iter
        self.gp_weight = conf.gp_weight
        self.use_cuda = conf.use_cuda
        self.num_steps = 0
        self.epochs = conf.epochs
        self.print_every = 50

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda() # 잘 됐는지 확인할 것

        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}

    def _critic_train_iteration(self, young_data, old_data, young_age, old_age):
        age_gap = get_age_gap(old_age, young_age)
        
        batch_size = self.batch_size
        young_data = Variable(young_data)
        old_data = Variable(old_data)
        young_age = Variable(young_age)
        old_age = Variable(old_age)
        age_gap = Variable(age_gap)
        
        if self.use_cuda:
            young_data = young_data.cuda()
            old_data = old_data.cuda()
            young_age = young_age.cuda()
            old_age = old_age.cuda()
            age_gap = age_gap.cuda()
            

        d_real = self.D(old_data, old_age)

        
        gen_old = self.sample_generator(young_data, age_gap)
        d_generated = self.D(gen_old, old_age) 

        gradient_penalty = self._gradient_penalty(old_data, gen_old, old_age)
        self.losses['GP'].append(gradient_penalty.item())

        self.d_optimizer.zero_grad()

        id_loss = identity_loss(young_data, gen_old, age_gap)

        gen_zero = self.sample_generator(young_data, )
        
        d_loss = d_generated.mean() - d_real.mean() + self.conf.gp_weight*gradient_penalty + self.conf.id_weight*id_loss
        d_loss.backward()

        self.d_optimizer.step()

        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, young_data, young_age, old_age):
        self.g_optimizer.zero_grad()
        batch_size = self.batch_size
        age_gap = get_age_gap(old_age, young_age)
        zero_gap = get_age_gap(young_age, young_age)
        young_data = Variable(young_data)
        young_age = Variable(young_age)
        old_age = Variable(old_age)
        age_gap = Variable(age_gap)
        zero_gap = Variable(zero_gap)

        if self.use_cuda:
            young_data = young_data.cuda()
            young_age = young_age.cuda()
            old_age = old_age.cuda()
            age_gap = age_gap.cuda()
            zero_gap = zero_gap.cuda()

        gen_old = self.sample_generator(young_data, age_gap)
        d_generated = self.D(gen_old, old_age)

        self_rec = self_rec_loss(young_data, zero_gap)
        g_loss = - d_generated.mean() + self.conf.self_rec_weight*self_rec
        g_loss.backward()
        self.g_optimizer.step()

        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data, old_age):
        batch_size = self.batch_size

        old_age = Variable(old_age)
        if self.use_cuda:
            old_age = old_age.cuda()

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, old_age)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            if data[3].shape[0]!=self.batch_size:
                continue
            self.num_steps += 1
            self._critic_train_iteration(data[0], data[1], data[2], data[3])

            if self.num_steps % self.critic_iter == 0:
                self._generator_train_iteration(data[0], data[2], data[3])

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iter:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif = True):
        if save_training_gif:
            import pandas as pd
            import random
            import numpy as np
            import imageio
            training_progress_images  = []

        if not os.path.isdir(self.conf.model_save_path):
            os.makedirs(self.conf.model_save_path, exist_ok=True)
            

        for ep in range(self.epochs):
            print("\nEpoch {}".format(ep + 1))
            self._train_epoch(data_loader)
            
            
            if save_training_gif:
                data =  next(iter(data_loader))
                age_gap = get_age_gap(data[3], data[2])
                if self.use_cuda:
                    
                    test_data = self.G(data[0].cuda(),age_gap.cuda()).cpu() + data[0] # mask + young data
                    
                    img_grid = make_grid(test_data)
                else:
                    img_grid = make_grid(self.G(data[0],age_gap))  + data[0]
                
                img_grid = np.transpose(img_grid.cpu().numpy(), (1,2,0))
                training_progress_images.append(img_grid.astype(np.uint8))

            if save_training_gif:
                imageio.mimsave('./result_example/training_{}_epochs.gif'.format(epochs),
                                training_progress_images)

            if (ep+1) % 30 == 0:
                
                torch.save(self.G.state_dict(), self.conf.model_save_path + 'G_model_{}epochs.pt'.format(ep+1))
                torch.save(self.D.state_dict(), self.conf.model_save_path + 'D_model_{}epochs.pt'.format(ep+1))

    def sample_generator(self, young_data, age_gap):
        generated_data = young_data + self.G(young_data, age_gap)
        return generated_data

    def sample(self, young_data, age_gap):
        generated_data = self.sample_generator(young_data, age_gap)
        return generated_data.data.cpu().numpy()[:, 0, :, :]
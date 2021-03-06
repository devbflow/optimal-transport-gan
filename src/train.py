import os
import time
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from data.Latent import *
from data.Gaussian import GaussianRing2D, GaussianNormal
from data.MNIST32 import MNIST32
#from data.Cifar import Cifar
from networks.DenseCritic import DenseCritic
from networks.DenseGenerator import DenseGenerator
from models.AssignmentModel import AssignmentModel

# set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class AssignmentTraining():

    def __init__(self,generator_net=None,
                 critic_net=None,
                 dataloader=None,
                 latent=None,
                 cost=None,
                 device='cpu'):
        self.dataloader = dataloader
        self.latent = latent
        self.critic = critic_net
        self.generator = generator_net
        self.cost = cost
        self.device = device
        self.experiment_name = self.dataloader.dataset.name + '_'\
                            + self.cost + '_' \
                            + self.latent.name + '_' \
                            + time.strftime("_%Y-%m-%d_%H-%M-%S_")
        self.log_path = os.path.join(os.path.dirname(os.getcwd()), "logs" + os.sep + self.experiment_name)
        self.model = AssignmentModel(self.dataloader,
                                     self.latent,
                                     self.generator,
                                     self.critic,
                                     self.cost,
                                     self.device)

    def train(self, n_critic_loops=None, n_main_loops=None):
        # init optimizers
        crit_opt = optim.RMSprop(self.critic.parameters(), lr=self.critic.lr)
        gen_opt = optim.RMSprop(self.generator.parameters(), lr=self.generator.lr)

        for ml in tqdm.tqdm(range(n_main_loops)): # generator train loops

            data_latent_ratio = len(self.dataloader.dataset) / self.latent.batch_size
            assign_loops = int(10 * data_latent_ratio * np.sqrt(ml / n_main_loops)) + 10
            #assign_loops = 1

            with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                for cl in crit_bar: # critic train loop

                    assign_arr, latent_samples, real_idcs = self.model.find_assignments_critic(assign_loops)
                    # assign_arr: torch.tensor with assign_arr[i] = number of assignments to real data point i
                    # latent_samples: list that contains
                    # real_idcs: list that contains the indices per batch as torch.tensors
                    self.model.train_critic(assign_arr, optimizer=crit_opt)
                    n_non_assigned = len(assign_arr) - torch.count_nonzero(assign_arr)
                    crit_bar.set_description(
                            "Step 1: Number of non assigned points " + str(n_non_assigned)
                            + ", Variance of perfect assignment " + str(torch.var(assign_arr)),
                            refresh=False)
            latent_samples = torch.vstack(tuple(latent_samples))
            real_idcs = torch.vstack(tuple(real_idcs)).flatten()

            self.model.train_generator(real_idcs, latent_samples, offset=16, optimizer=gen_opt)

            # images for tensorboard (TODO: unimplemented right now)
            #if ml % 50 == 1:

    def log_data(self, main_loop, max_loop):
        # TODO
        pass

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    dataset = GaussianNormal(batch_size=100, device=device)
    #dataset = GaussianRing2D(batch_size=50, radius=5, N=10, num_data=1000, device=device)
    #dataset = Cifar()
    #dataset = MNIST32(root='./data', download=True)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=False)
    #print(len(dataloader))
    #for idx, sample in enumerate(dataloader):
    #    print(sample[0], sample[1])
    #    print(idx)
    #    break


    latent = GaussianLatent(shape=2, batch_size=dataloader.batch_size*10)
    critic = DenseCritic(name="critic", lr=1e-4, layer_dim=64, xdim=np.prod(dataset.data_shape))
    generator = DenseGenerator(name="generator", lr=5e-5, layer_dim=64, xdim=np.prod(dataset.data_shape))

    assignment = AssignmentTraining(dataloader=dataloader,
                                    latent=latent,
                                    critic_net=critic,
                                    generator_net=generator,
                                    cost="square",
                                    device=device)

    assignment.train(n_main_loops=200, n_critic_loops=10)


if __name__ == "__main__":
    main()

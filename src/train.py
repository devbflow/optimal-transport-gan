import os
import time
import tqdm

import torch
import numpy as np

from data.Latent import *
from data.Gaussian import GaussianRing2D
from data.Mnist import Mnist32
from data.Cifar import Cifar
from networks.DenseCritic import DenseCritic
from networks.DenseGenerator import DenseGenerator
from models.AssignmentModel import AssignmentModel

class AssignmentTraining():

    def __init__(self,generator_net=None,
                 critic_net=None,
                 dataset=None,
                 latent=None,
                 cost=None):
        self.dataset = dataset
        self.latent = latent
        self.critic = critic_net
        self.generator = generator_net
        self.cost = cost
        self.experiment_name = self.dataset.name + '_'\
                            + self.cost + '_' \
                            + self.latent.name + '_' \
                            + time.strftime("_%Y-%m-%d_%H-%M-%S_")
        self.log_path = os.path.join(os.path.dirname(os.getcwd()), "logs" + os.sep + self.experiment_name)
        self.model = AssignmentModel(self.dataset,
                                     self.latent,
                                     self.generator,
                                     self.critic,
                                     self.cost)

    def train(self, n_critic_loops=None, n_main_loops=None):
        #n_non_assigned = self.latent.batch_size
        for ml in tqdm.tqdm(range(n_main_loops)):
            data_latent_ratio = self.dataset.dataset_size / self.latent.batch_size
            assign_loops = int(10 * data_latent_ratio * np.sqrt(ml / n_main_loops)) + 10
            #assign_loops = 2
            with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                for cl in crit_bar:
                    assign_arr, latent_samples, real_idcs = self.model.find_assignments_critic(assign_loops)
                    # assign_arr: torch.tensor with assign_arr[i] = number of assignments to real data point i
                    # latent_samples: list that contains
                    # real_idcs: list that contains the indices per batch as torch.tensors

                    self.model.train_critic(assign_arr, optimizer=None)
                    n_non_assigned = len(assign_arr) - np.count_nonzero(assign_arr)
                    crit_bar.set_description(
                            "Step 1: Number of non assigned points " + str(n_non_assigned)
                            + ", Variance of perfect assignment " + str(np.var(assign_arr.detach().numpy())),
                            refresh=False)
            latent_samples = np.vstack(tuple(latent_samples))
            real_idcs = np.vstack(tuple(real_idcs)).flatten()

            self.model.train_generator(real_idcs, latent_samples, offset=16, optimizer=None)

            # images for tensorboard (TODO: unimplemented right now)
            #if ml % 50 == 1:

    def log_data(self, main_loop, max_loop):
        # TODO
        pass

def main():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            dev = torch.device('cuda')
        #else:
    else:
        dev = torch.device('cpu')

    #dataset = GaussianRing2D(batch_size=16, radius=.5, N=10, num_data=16)
    #dataset = Mnist32(batch_size=16, dataset_size=50000)
    dataset = Cifar(batch_size=16, dataset_size=50000)
    latent = MultiGaussianLatent(shape=250, batch_size=16, N=100)
    critic = DenseCritic(name="critic", lr=1e-4, layer_dim=128, xdim=32*32*3)
    generator = DenseGenerator(name="generator", lr=5e-5, layer_dim=64, xdim=32*32*3)
    #print("generator initial params = ", [p for p in generator.parameters()])
    #print("critic initial params = ", [p for p in critic.parameters()])

    assignment = AssignmentTraining(dataset=dataset,
                                    latent=latent,
                                    critic_net=critic,
                                    generator_net=generator,
                                    cost="square")

    assignment.train(n_main_loops=200, n_critic_loops=5)
    #print("trained generator params = ", [p for p in assignment.generator.parameters()], [p for p in assignment.model.generator.parameters()])
    #print("trained critic params = ", [p for p in assignment.critic.parameters()], [p for p in assignment.model.critic.parameters()])


if __name__ == "__main__":
    main()

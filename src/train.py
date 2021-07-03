import os
import time
import tqdm

import torch
import numpy as np

from data.Latent import *
from data.Gaussian import GaussianRing2D
from networks.DenseCritic import DenseCritic
from networks.DenseGenerator import DenseGenerator
from models.AssignmentModel import AssignmentModel

# TODO: import classes from directory

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

            self.model.train_generator(real_idcs, latent_samples, offset=4, optimizer=None) # TODO change offset to 16

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

    dataset = GaussianRing2D(batch_size=4, radius=5, N=2, num_data=32)
    latent = MultiGaussianLatent(shape=2, batch_size=4, N=200)
    critic = DenseCritic(name="critic", lr=1e-4, layer_dim=1024, xdim=2)
    generator = DenseGenerator(name="generator", lr=5e-5, layer_dim=512, xdim=2)

    assignment = AssignmentTraining(dataset=dataset,
                                    latent=latent,
                                    critic_net=critic,
                                    generator_net=generator,
                                    cost="square")

    assignment.train(n_main_loops=10, n_critic_loops=5)

if __name__ == "__main__":
    main()

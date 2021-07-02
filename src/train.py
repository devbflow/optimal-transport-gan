import os
import time
import tqdm

import torch
import numpy as np


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
        self.n_non_assigned = self.latent.batch_size
        for ml in tqdm.tqdm(range(n_main_loops)):
            data_latent_ratio = self.dataset.dataset_size / self.latent.batch_size
            assign_loops = int(10 * data_latent_ratio * np.sqrt(main_loop/n_main_loops)) + 10
            with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                for cl in crit_bar:
                    assign_arr, latent_sample, real_idx = self.model.find_assignments_critic(assign_loops)
                    self.model.train_critic(assign_arr, optimizer=None)
                    self.n_non_assigned = len(assign_arr) - np.count_nonzero(assign_arr)
                    crit_bar.set_description(
                            "Step 1: Number of assigned points " + str(self.n_non_assigned)
                            + ", Variance of perfect assignment " + str(np.var(assign_arr)),
                            refresh=False)
            latent_sample = np.vstack(tuple(latent_sample))
            real_idx = np.vstack(tuple(real_idx)).flatten()

            self.model.train_generator(real_idx, latent_sample, offset=16, optimizer=None)

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
        else:
            # TODO multiple gpus
    else:
        dev = torch.device('cpu')
    #TODO: insert values, move to gpu
    assignment = AssignmentTraining(dataset=None,
                                    latent=None,
                                    critic_net=None,
                                    generator_net=None,
                                    cost=None)
    assignment.to(dev)
    assignment.train(n_main_loops=200, n_critic_loops=10)

if __name__ == "__main__":
    main()

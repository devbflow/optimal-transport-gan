import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np

class AssignmentModel:

    def __init__(self, dataloader=None, latent=None, generator=None, critic=None, cost=None, A_couples=None, A_cost=None):
        ## Variables ##
        self.cost = cost
        self.dataset = dataloader.dataset
        self.dataloader = dataloader
        self.latent = latent
        self.generator = generator
        self.critic = critic
        self.A_couples = A_couples
        self.A_cost = A_cost

        #self.power_factors = (0.0448, 0.2856)

        ## Architecture ##
        # generates a random latent batch on call, is function object
        self.gen_latent_batch = self.latent.sample

        if self.cost == "ssim":
            self.find_couples = self.find_couples_unlimited_ssim
            self.gen_cost = self.assign_gen_cost_ssim
        elif self.cost == "psnr":
            self.find_couples = self.find_couples_unlimited_psnr
            self.gen_cost = self.assign_gen_cost_psnr
        elif self.cost == "square":
            self.find_couples = self.find_couples_unlimited_square
            self.gen_cost = self.assign_gen_cost_square
        else:
            raise ValueError("cost must be one of ['ssim', 'psnr', 'square']")


    def find_assignments_critic(self, assign_loops=100):
        num_batches = len(self.dataset) // self.dataset.batch_size
        assign_arr = torch.zeros((len(self.dataset),))
        latent_sample_list, real_idx_list = [], []

        for _ in range(assign_loops):
            latent_points = self.gen_latent_batch(self.latent.batch_size)
            latent_points.to(next(self.generator.parameters).device) # move to same device as generator
            generated_batch = self.generator(latent_points)
            '''
            ### New start ###
            for batch_idx, sample in enumerate(self.dataloader):
                # get batch from sample, sample is tuple of (batch, labels)
                real_batch, labels = sample
                indices = np.arange(self.dataloader.batch_size) * batch_idx
                if batch_idx != 0:
                    #print(indices, curr_best)
                    indices = np.vstack((curr_best, indices))
                    # construct sub-dataset from given indices to combine previous best with current indices
                    subset = Subset(self.dataloader.dataset, indices)
                    subset_loader = DataLoader(subset, len(subset))
                    real_batch = subset_loader.next()[0]
                    print(real_batch.shape)
                real_batch = real_batch.reshape(-1, np.prod(real_batch.shape[1:]))
                best = self.find_couples(real_batch, generated_batch)
                curr_best = indices[best]

            #assert len(curr_best.shape) == 1
            assign_c = torch.tensor(curr_best).unsqueeze(dim=1)
            latent_samples.append(latent_points)
            real_indcs.append(assign_c)
            idx_values = torch.unique(assign_c, return_counts=True)
            assign_arr[idx_values[0]] += idx_values[1]
        return assign_arr, latent_samples, real_indcs

            ### New end ###
            '''
            for b in range(num_batches):
                indices = np.arange(b * self.dataloader.batch_size, (b+1)* self.dataloader.batch_size)
                # add the current best from previous batch(es) into comparison
                if b == 0:
                    all_idx = indices
                else:
                    #print("current_best = ", current_best, current_best.shape)
                    #print("indices = ", indices, indices.shape)
                    all_idx = np.concatenate([current_best, indices], axis=0)
                    #print("all_idx = ", all_idx, all_idx.shape)
                    #break
                # returns indices of best couples from current batch
                best = self.find_couples(real_batch=self.dataset[all_idx], generated_batch=generated_batch)
                current_best = all_idx[best]

            assign_c = torch.tensor(current_best).reshape(-1, 1)
            latent_sample_list.append(latent_points)
            real_idx_list.append(assign_c)
            idx_value = torch.unique(assign_c, return_counts=True)
            assign_arr[idx_value[0]] += idx_value[1]
        return assign_arr, latent_sample_list, real_idx_list

    # A_w(X), X = real_samples
    def assign_critic_cost(self, assign_samples, n_assign):
        crit_assign = self.critic(assign_samples)
        crit_assign_weigthed = torch.mul(n_assign, assign_samples)
        assign_w_n_ratio = torch.sum(crit_assign_weigthed) / torch.sum(n_assign)
        dataset_mean = torch.mean(self.critic(self.dataset.data))
        crit_cost = -(assign_w_n_ratio - dataset_mean)
        return crit_cost

    ### Square ###
    def find_couples_unlimited_square(self, real_batch, generated_batch):
        #print(real_batch.shape, generated_batch.shape)
        # use broadcasting to get a matrix with all fakes - each real
        z = torch.unsqueeze(generated_batch, dim=1) - real_batch
        # square distance for matrix
        norm_mat = 0.1 * torch.square(torch.linalg.norm(z, dim=2))
        dist = torch.transpose(self.critic(real_batch), 0, 1) + norm_mat
        # return real positions where distance is minimal
        couples = torch.argmin(dist, dim=1)
        return couples

    # cost(X, G(z)), X = real samples, G(z) = generated samples
    def assign_gen_cost_square(self, real_batch, generated_batch):
        diff_batch = generated_batch - real_batch
        gen_cost = torch.mean(torch.square(torch.linalg.norm(diff_batch, dim=1)))
        return gen_cost

    ### SSIM ###
    def find_couples_unlimited_ssim(self, real_batch, generated_batch):
        # TODO: external code/pytorch ignite version?
        pass

    def assign_gen_cost_ssim(self, real_batch, generated_batch):
        #TODO
        pass

    ### PSNR ###
    def find_couples_unlimited_psnr(self, real_batch, generated_batch):
        # TODO: external code?
        pass

    def assign_gen_cost_psnr(self, real_batch, generated_batch):
        #TODO
        pass

    ### Train Critic/Generator ###

    def train_critic(self, assign_arr, optimizer=None):
        if optimizer is None:
            optimizer = optim.RMSprop(self.critic.parameters(), lr=self.critic.lr)
        assign_idx_local = np.nonzero(assign_arr)
        assign_samples = self.dataset.data[assign_idx_local]
        n_assign = assign_arr[assign_idx_local]
        # train step
        crit_cost = self.assign_critic_cost(assign_samples, n_assign)
        optimizer.zero_grad()
        crit_cost.backward()
        optimizer.step()


    def train_generator(self, real_idcs, latent_samples,  offset=2000, optimizer=None):
        if optimizer is None:
            optimizer = optim.RMSprop(self.generator.parameters(), lr=self.generator.lr)

        cost = []
        latent_samples = torch.tensor(latent_samples)
        for c_idx in range(0, int(len(real_idcs) - offset + 1), int(offset)):
            real_batch = self.dataset.data[real_idcs[c_idx:c_idx+offset]]
            generated_batch = self.generator(latent_samples[c_idx:c_idx+offset])
            # train step
            gen_cost = self.gen_cost(real_batch, generated_batch)
            optimizer.zero_grad()
            gen_cost.backward()
            optimizer.step()
            cost.append(gen_cost)
        print("The transportation distance is", np.sqrt(np.average([c.detach().numpy() for c in cost])))

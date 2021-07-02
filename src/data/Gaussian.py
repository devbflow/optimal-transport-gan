import torch
import numpy as np

class GaussianRing2D(DatasetABC):

    def __init__(self, batch_size, radius, N=10, num_data=1000):
        super().__init__()
        self.data = generate_data(radius, N=N, num_data=num_data, tensor=True)
        self.name = "GaussianRing"
        self.batch_size = batch_size
        self.shape = (2,)


    @staticmethod
    def generate_data(radius, N=10, num_data=1000, std=.1, origin=(0,0), tensor=False, seed=None):
        """Generates 2-dimensional Gaussians on a ring with given number of data points and dimensionality.
        Means of the Gaussians are located on the circle with given radius around the origin.

        Parameters
        ----------
            radius : {int, float}
                radius of the circle around the origin
            N : int
                number of Gaussians to generate, default is 10
            num_data : int
                number of data points to be generated for each Gaussian, default is 1000
            std : {int, float, numpy.ndarray}
                standard deviation of the Gaussians, can be float or ndarray,
                if float, all Gaussians have the same std,
                if ndarray, Gaussians have their own std, must have dim N
                default is 1.
            origin : array_like
                origin of the circle, default is (0,0)
            tensor : bool
                if True, output is a torch.tensor instead of numpy.ndarray, default is False
            seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
                seed for the RNG for repeating experiments, default is None
        Output
        ------
            gaussians : {numpy.ndarray, torch.tensor}
                ndarray/tensor of shape (N, num_data, 2)
        """
        # check types of inputs
        if type(std) is float or type(std) is int:
            std = np.full((N, 2), float(std))
        elif type(std) is not np.ndarray:
            raise ValueError("Parameter 'std' is not int, float or np.ndarray")

        if type(radius) is not float:
            if type(radius) is not int:
                raise ValueError("Parameter 'radius' is not float or int")
            radius = float(radius)

        # define variables
        dim = 2
        gaussians = np.zeros((N, num_data, dim))
        origin = np.array(origin)
        centers = np.zeros((N, dim))
        rng = np.random.default_rng(seed)

        # calculate Gaussian centers on circle and generate data for each
        for i in range(N):
            centers[i,0] = radius * np.cos((i*2*np.pi)/N) + origin[0]
            centers[i,1] = radius * np.sin((i*2*np.pi)/N) + origin[1]
            # generate num_data points for each of the Gaussians
            data = rng.normal(centers[i], std[i], (num_data, dim))
            gaussians[i] = data

        # return torch.Tensor instead of numpy.ndarray if desired
        if tensor:
            gaussians = torch.from_numpy(gaussians)
        return gaussians

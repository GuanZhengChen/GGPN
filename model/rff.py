import numpy as np
import torch
import sys
# from misc_utils import set_random_seed
# from gaussian_exact import GaussianKernel

class RFF(torch.nn.Module):
    def __init__(self, num_samples, num_dims, kernel=None, device='cpu'):
        super(RFF, self).__init__()
        self.num_samples = num_samples  # number of rff features
        self.kernel = kernel
        self.num_dims = num_dims # dimension of the original input
        self.device = device
        self.w = torch.from_numpy(np.random.normal(scale=1.0, 
            size=(int(self.num_samples), int(self.num_dims)))).to(torch.float32)
        self.w = torch.nn.Parameter(self.w).to(self.device)
        self.b = torch.from_numpy(np.random.uniform(low=0.0, high=2.0 * np.pi, size=(int(self.num_samples), 1))).to(torch.float32)
        # self.b = torch.nn.Parameter(self.b).to(self.device)

    def forward(self, x_i, x_j):
        rff_x1 = self.get_featurize(x_i)
        rff_x2 = self.get_featurize(x_j)
        k = rff_x1.mul(rff_x2)
        k = torch.sum(k, dim=1)
        return k

    # @torch.no_grad()
    def get_featurize(self, x):
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        self.input = torch.mm(self.w, torch.transpose(x, 0, 1))
        
        self.feat = float(np.sqrt(2/float(self.num_samples) ) ) * torch.cat([torch.cos(self.input), torch.sin(self.input)], dim=0)
        return torch.transpose(self.feat, 0, 1)

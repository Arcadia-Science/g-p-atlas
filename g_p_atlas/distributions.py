import torch
from torch.autograd import Variable


class genetic_allelic_state_distribution():
 def __init__(self,n_allelic_states,n_loci):
  real_cat_dist = (1.0/n_alleles)*np.ones((n_locs,n_alleles))
  real_cat_dist = torch.distributions.OneHotCategorical(torch.tensor(real_cat_dist))
  z_real_cat = Variable(torch.flatten(real_cat_dist.sample(sample_shape=torch.Size([batch_size])),start_dim=1))
  z_real_cat = z_real_cat.to(device)
  z_real_cat = z_real_cat.float()

import torch

from torch import nn
from torch_geometric.utils import to_dense_adj

from torch.nn.functional import relu
import matplotlib.pyplot as plt
import normflows as nf

import math
from torch.nn.functional import relu

from tqdm import tqdm
from utils.printing_utils import printd
from utils.plotting import *

from utils.utils import get_prob_graph

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}



def get_cov(e_vals : torch.tensor, rot : torch.tensor):
    '''
    :param: e_vals: eigenvalues of the covariance matrix (inflation of the axes)
    :param: rot: the angle by which to rotate the covariance matrix

    :return: the covariance matrix
    '''
    
    def rot_2d(theta):
        return torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                         [torch.sin(theta), torch.cos(theta)]])
    
    cov_unrotated = torch.diag(e_vals)
    rot_mat = rot_2d(rot)
    cov = rot_mat@cov_unrotated@rot_mat.T
    return cov


def identity(x):
    return x

def trans_hyp2d(x):
    '''
    this function returns the coordinates in the EUCLIDIAN plane
    x is the hyperbolic coordinates. 
    '''
    if len(x.shape) == 1:
        # PATCH: the way to use functorch is to assume that the rows are independent and so you can't mix rows! you get a tensor that is everything but the rows and you do operations on this new rowless tensor.
        '''for jac '''
        f0 = x[1]*torch.exp(x[0])
        f1 = x[1]*torch.exp(-x[0])
        return torch.hstack([f0, f1])
    else:  
        '''for loss'''
        f0 = x[:,1]*torch.exp(x[:,0])
        f1 = x[:,1]*torch.exp(-x[:,0])
        feats = torch.cat((f0.unsqueeze(dim=1), f1.unsqueeze(dim=1)), dim=1)
    return feats
                            


def get_trans_identity():
    return lambda x: x


def get_trans_linear(mean,cov2):
    S = torch.linalg.cholesky(cov2)
    def f(x):
        feats = x@S.T+mean
        return feats
    return f

def get_trans_hyp2d():
    return trans_hyp2d

def get_composed_trans(f1,f2):
    def f(x):
        return f1(f2(x))
    return f


# 88""Yb 888888  dP""b8 
# 88__dP 88__   dP   `" 
# 88"Yb  88""   Yb  "88 
# 88  Yb 888888  YboodP         


# 888888 88""Yb    db    88b 88 .dP"Y8 
#   88   88__dP   dPYb   88Yb88 `Ybo." 
#   88   88"Yb   dP__Yb  88 Y88 o.`Y8b 
#   88   88  Yb dP""""Yb 88  Y8 8bodP'          
                                                                                                                                                                                                        
def relu_transform(x, grad, lr, node_mask, cutoff=0.0):
    '''
    this function is a transformation that is a combination of the relu and the identity. 
    the relu is applied to the nodes that are true in the node mask. 
    '''
    return relu(x + lr*grad*node_mask.unsqueeze(-1) - cutoff) + cutoff

def relu_lightcone(xt, grad, lr, node_mask, cutoff=0.0):
    '''does relu to the point update on the lightcone. coordinates are rotated-relu-rotated back'''
    #todo: replace the first lines of code with the function uv_from_xt
    rotation_mat = torch.tensor([[1.0, -1.0], [1.0, 1.0]], device=xt.device)
    num_feats = xt.shape[1] 
    xs = xt[:, :num_feats//2]
    ts = xt[:, num_feats//2:]
    dxs = grad[:, :num_feats//2]
    dts = grad[:, num_feats//2:]
    
    xt_cube = torch.cat([xs.unsqueeze(-1), ts.unsqueeze(-1)], dim=2)
    dxdt_cube = torch.cat([dxs.unsqueeze(-1), dts.unsqueeze(-1)], dim=2)
    #* 
    uv = torch.einsum('ij,klj->kli', rotation_mat, xt_cube)

    dudv = torch.einsum('ij,klj->kli', rotation_mat, dxdt_cube)
    
    chopped_uv = relu(uv + lr*dudv*node_mask.unsqueeze(-1).unsqueeze(-1))
    #? TESTED: adding dudv only alters the nodes that are true in the node mask -> 
    #? torch.unique(torch.where(chopped_uv - uv == 0)[0]) == torch.where(node_mask == 0)[0]
    chopped_xt = 0.5*torch.einsum('ij,klj->kli', rotation_mat.T, chopped_uv)
    rearange_chopped = torch.cat([chopped_xt[:, :, 0], chopped_xt[:, :, 1]], dim=1)
    #? TESTED: when there is no need for relu, rearange_chopped == xt+ lr*grad

    return rearange_chopped

def relu_lightcone_pts(xt):
    rotation_mat = torch.tensor([[1.0, -1.0], [1.0, 1.0]], device=xt.device)
    num_feats = xt.shape[1] 
    xs = xt[:, :num_feats//2]
    ts = xt[:, num_feats//2:]
    
    xt_cube = torch.cat([xs.unsqueeze(-1), ts.unsqueeze(-1)], dim=2)
    #* 
    uv = torch.einsum('ij,klj->kli', rotation_mat, xt_cube)
    
    chopped_uv = relu(uv)
    #? TESTED: adding dudv only alters the nodes that are true in the node mask -> 
    #? torch.unique(torch.where(chopped_uv - uv == 0)[0]) == torch.where(node_mask == 0)[0]
    chopped_xt = 0.5*torch.einsum('ij,klj->kli', rotation_mat.T, chopped_uv)
    rearange_chopped = torch.cat([chopped_xt[:, :, 0], chopped_xt[:, :, 1]], dim=1)
    return rearange_chopped

def uv_from_xt(xt):
    '''takes xt that are arranged [xs..., ts...] and returns uv in 3d tensor form [u1,v1, u2,v2, ...]'''
    rotation_mat = 1/math.sqrt(2)*torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    num_feats = xt.shape[1] 
    xs = xt[:, :num_feats//2]
    ts = xt[:, num_feats//2:]
    
    xt_cube = torch.cat([xs.unsqueeze(-1), ts.unsqueeze(-1)], dim=2)
    # xs is xt_cube[:,:,0]

    uv = torch.einsum('ij,klj->kli', rotation_mat, xt_cube)
    # us are uv[:,:,0]?
    return uv


# 88""Yb 88""Yb 88  dP"Yb  88""Yb 
# 88__dP 88__dP 88 dP   Yb 88__dP 
# 88"""  88"Yb  88 Yb   dP 88"Yb  
# 88     88  Yb 88  YbodP  88  Yb 


def train_prior(prior, x, optimizer, n_iter=1000, verbose=False, show_iter=100):
    '''
    this function optimizes the model. 
    noise is added to the feature optimization and to the input nodes when training.
    ''' 
    #! check to see if i need the next line...
    # graph.x = graph.x.detach().requires_grad_(True)

    show_iter = min(show_iter, n_iter)
    prior.model.train()
    losses = []
    loss = -prior.forward_ll(x)

    for i in tqdm(range(n_iter), desc="prior opt"):
        optimizer.zero_grad()

        loss = -prior.forward_ll(x)
        losses.append(loss.item())

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        else:
            raise ValueError('loss is nan or inf at iter ' + str(i))
        # if verbose:
        #         if (i+1) % show_iter == 0:
        #             print(f'optimize net: iter {i+1}/{n_iter}, loss = {loss.item()}')
        #             plot_optimization_stage(i, n_iter, prior, graph, losses, lorenz_fig_lims,calling_function_name='train_prior')
    return losses

# 888888 88      dP"Yb  Yb        dP .dP"Y8 
# 88__   88     dP   Yb  Yb  db  dP  `Ybo." 
# 88""   88  .o Yb   dP   YbdPYbdP   o.`Y8b 
# 88     88ood8  YbodP     YP  YP    8bodP' 

class HyperbolicInflate(nf.flows.Flow):
    '''this inflates the features that are next to 0 so that they are more spaced out, while only shifting the features that are close to 1'''
    def __init__(self):
        super().__init__()

    def inverse(self, z):
        #todo: this should either be relu lightcone or relu depending on lorenz...
        z = relu(z)
        z = torch.sign(z)*torch.sqrt(z**2 + torch.sign(z)*z + 1e-8)
        # pre_log_det = (2*z+1)/torch.sqrt(z**2 + z)
        log_det_pre_sum = torch.log(2*z + 1) - 0.5*torch.log(z**2 + z)
        log_det = torch.sum(log_det_pre_sum, dim=1)
        return z, log_det
    def forward(self, z):
        pass
    
class RootInflate(nf.flows.Flow):
    '''this inflates the features that are next to 0 so that they are more spaced out, while only shifting the features that are close to 1'''
    def __init__(self):
        super().__init__()

    def inverse(self, z):
        #todo: this should either be relu lightcone or relu depending on lorenz...
        z = relu(z)
        z = torch.sign(z)*torch.sqrt(torch.sign(z)*z + 1e-8)
        # pre_log_det = (2*z+1)/torch.sqrt(z**2 + z)
        log_det_pre_sum = -torch.log(2) - 0.5*torch.log(torch.sign(z)*z)
        log_det = torch.sum(log_det_pre_sum, dim=1)
        return z, log_det
    def forward(self, z):
        pass


# 8b    d8  dP"Yb  8888b.  888888 88     .dP"Y8 
# 88b  d88 dP   Yb  8I  Yb 88__   88     `Ybo." 
# 88YbdP88 Yb   dP  8I  dY 88""   88  .o o.`Y8b 
# 88 YY 88  YbodP  8888Y"  888888 88ood8 8bodP' 
#todo: another thing to try is to to omit all of the dyads between the test nodes
class RealNVP(nn.Module):
    '''
    this is an implementation of the real NVP transformation from the normflows package examples. 
    regularized by noise amp which is added to the input
    '''
    def __init__(self, 
                 in_out_dim=2, 
                 hidden_dim=64,
                 num_coupling_blocks = 32, 
                 num_layers_mlp=2,
                 attr_opt=False,
                 activation=nn.ReLU, 
                 inflation_flow_name=None, 
                 device=torch.device('cpu')):
        
        super().__init__()
        self.attr_opt = attr_opt
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.num_layers_mlp = num_layers_mlp
        self.num_coupling_blocks = num_coupling_blocks
        #! should trainable be false? i think so... in the example they were changing the gaussian to be something else
        self.base = nf.distributions.base.DiagGaussian(self.in_out_dim, trainable=True)

        self.flows = []

        for _ in range(self.num_coupling_blocks):
            ''' Q:  why mlp input and output dims like this? 
                A:  coupling block splits z = [z1(input to net), z2 (transformed by net output)]. 
                    input is math,ceil(...) because if dim(z) odd, dim(z1) = dim(z2) + 1.
                    output is 2*(dim(z)//2) because the output is concat (s(z1), t(z1)) and are both in the dim(z2).
            '''
            layer_dims = [math.ceil(self.in_out_dim/2)] + [self.hidden_dim]*self.num_layers_mlp + [2*(self.in_out_dim//2)]
            param_map = nf.nets.MLP(layer_dims, init_zeros=True)
            # Add flow layer. param map is a mapping from the x_A parameters
            
            self.flows.append(nf.flows.AffineCouplingBlock(param_map))
            
            # Swap dimensions
            #! does permute do the same permutation every forward pass?
            self.flows.append(nf.flows.Permute(self.in_out_dim, mode='swap')) 
            
        # INFLATION AS FINAL LAYER (since convention is that latent space is last)    
        if inflation_flow_name is not None:
            if inflation_flow_name == 'hyperbolic':
                inflation_flow = HyperbolicInflate()
            self.flows.append(inflation_flow)
            # Construct flow model  

        self.model = nf.NormalizingFlow(self.base, self.flows)
        
        self.model.to(device)
        self.eval()

    def backward_xz(self, x, noise_amp, limit_radius=10):
        '''x(original space) -> coordinate change -> z(perfect gaussian)
           args:
              x: the input
              noise_amp: the amplitude of the noise added to the input (regularization)
              limit_radius: can't go larger than this radius
            '''

        if self.training:
            # x_norm = x.norm(dim=1, keepdim=True)
            #todo: we can do what we used to do, only to insert the noise AFTER we do inflation!!
            z = x + noise_amp*torch.randn_like(x)
        else:
            z = x
        #todo: the first flow make a hyperbolic maybe somewhere in the definitions of the flow

        log_det_tot = torch.zeros(len(x), device=x.device)
        #* for now the plan is to take the nodes that flow too far away and keep them at the farthermost point
        move_forward_mask = torch.ones(x.shape[0], device=x.device).bool()
        for i in range(len(self.flows) - 1, -1, -1):
            '''at every step of the flow, we check which of the nodes it threw too far away and we keep them with the last value they had before they went too far away. '''
            #! maybe it's the definition of the prior?
            z_out, log_det = self.flows[i].inverse(z)
            z_out = torch.clamp(z_out, -1000*limit_radius, 1000*limit_radius)
            #* don't have to stop z completely maybe? no need to update the mask
            move_forward_mask = move_forward_mask *((z.abs() < limit_radius).all(dim=1))
            #!debugVV
            # z_old = z.clone()
            #! debug ^^
            z = ~move_forward_mask.unsqueeze(1)*z + move_forward_mask.unsqueeze(1)*z_out
            #! debugVV
            # if (z > limit_radius).any():
                # raise ValueError('z is too large')
                # a=0
                # printd(f'{(z > limit_radius).any(dim=1).sum()=}')
            #!debug ^^
            
            log_det_tot += move_forward_mask*log_det
            
            #!debug
            # if z_out.isinf().any():
            #     raise ValueError('z is nan')
            #!debug^^
                
        #TODO: for some reason the prior gives nan here. 
        #TODO: return to the link pred model, there is are other todos. mainly, download the anomaly models and insert anomalies into your graphs.
        return z, log_det_tot
   

    def forward_ll(self, x, noise_amp=0.0,sum=True):
        '''x -> coordinate change -> log likelihood of samples in the new coordinates with the standard normal distribution
        this is supposed to be the log_prob function that comes with the package but i put in some modifications.
        '''
        #
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        z, log_q = self.backward_xz(x, noise_amp)
        
        log_prob = self.model.q0.log_prob(z) + log_q
        if sum:
            return torch.sum(log_prob)
        else:
            return(log_prob)
    
    def inverse_and_log_det(self, x):
        '''could this be the derivative wrt to x?'''
        return self.model.inverse_and_log_det(x)

    def plot_latant_dist(self):
        ''' this would plot the distribution in the latant space, where zs are. we take a grid of zs and calculate the probability in that space.'''
        grid_size = 200
        xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        # zz is a list of all of the points xy on the grid torch.Size([40000, 2])

        self.model.eval()
        log_prob = self.model.log_prob(zz).to('cpu').view(*xx.shape)
        self.model.train()
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(5, 5))
        plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    def save_weights(self, dir_path):
        'save state with the input and hidden dimensions so we know what we are loading'
        params = self.model.state_dict()
        torch.save(params, dir_path + '/prior_weights_' + str(self.in_out_dim) + '_' + str(self.hidden_dim) +'_' + str(self.num_coupling_blocks) +'.pth')


class MinValueScheduler:
    def __init__(self, scheduler, min_lr):
        self.scheduler = scheduler
        self.min_lr = min_lr

    def get_lr(self):
        return max(self.scheduler.get_lr(), self.min_lr)

    def step(self):
        self.scheduler.step()
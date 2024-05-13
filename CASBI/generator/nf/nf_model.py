import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import time
import os
import re
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split 



class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) to be obtain parameters of Coupling layers
    
    Parameters
    ----------
    n_input : int 
        Number of input neurons, depend on the dimensions of the input data. 
    n_output : int 
        Number of output neurons, depend on the number of parameters needed for the Coupling layers.
    n_hidden : int
        Number of hidden neurons in each layer.
    n_layers : int
        Number of layers in the network.
    neg_slope : float
        Negative slope for the leaky ReLU activation function.
    
    Returns
    -------
    None
    """

    def __init__(self, n_input, n_output, n_hidden, n_layers=4, neg_slope=0.2) -> None:
        super().__init__()
        ins = torch.ones(n_layers)*n_hidden
        ins[0] = n_input
        outs = torch.ones(n_layers)*n_hidden
        outs[-1] = n_output
        Lin_layers = list(map(nn.Linear, ins.type(torch.int), outs.type(torch.int)))
        ReLu_layers = [nn.LeakyReLU(neg_slope) for _ in range(n_layers)]
        self.network = nn.Sequential(*itertools.chain(*zip(Lin_layers, ReLu_layers)))
        # self.network.apply(init_weights)
    
    def forward(self, x):
        return self.network(x.float())

    
class GLOW_conv(nn.Module):
    def __init__(self, n_dim) -> None:
        super().__init__()
        self.n_dim = n_dim

        #Initialize W as orthogonal matrix and decompose it into P, L, U, the learned parameters
        W_initialize = nn.init.orthogonal_(torch.randn(self.n_dim, self.n_dim))
        P, L_, U_ = torch.linalg.lu(W_initialize)

        #P not changed (no grad) but it needs to be stored in the state_dict
        self.register_buffer("P", P)

        # Declare as model parameters
        #Diagonal of U sourced out to S
        S_ = torch.diagonal(U_)
        self.S = nn.Parameter(S_)
        self.L = nn.Parameter(L_)
        #Declare with diagonal 0s, without changing U_ and thus S_
        self.U = nn.Parameter(torch.triu(U_, diagonal=1))

    def _get_W_and_logdet(self):
        #Make sure the pieces stay in correct shape as in GLOW
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.n_dim).to(self.L.device))
        U = torch.triu(self.U, diagonal=1)
        S = torch.diag(self.S)
        
        W = self.P@L@(U+S)
        logdetW = torch.sum(torch.log(torch.abs(self.S)))

        return W, logdetW  
    
    # Pass condition as extra argument, that is not used in the convolution
    #it stayes untouched, does not get permuted with values that
    #will be transformed
    def forward(self, x, x_condition):
        W, logdetW = self._get_W_and_logdet()
        y = x.float()@W
        return y, logdetW
    
    def backward(self, y, x_condition):
        W, logdetW_inv = self._get_W_and_logdet()
        #Just a minus needed
        logdetW_inv = -logdetW_inv
        W_inv = torch.linalg.inv(W)
        x = y.float()@W_inv
        return x, logdetW_inv

class AffineCoupling(nn.Module):
    """
    Affine Coupling layer for conditional normalizing flows.

    Args:
        dim_notcond (int): Dimension of the input not conditioned part.
        dim_cond (int): Dimension of the input conditioned part.
        network (nn.Module): Network architecture to use for the affine coupling layer.
        network_args (tuple): Additional arguments to pass to the network architecture.

    Attributes:
        dim_notcond (int): Dimension of the input not conditioned part.
        dim_cond (int): Dimension of the input conditioned part.
        net_notcond (nn.Module): Network for the not conditioned part.
        net_cond (nn.Module): Network for the conditioned part.
    """

    def __init__(self, dim_notcond, dim_cond, network=MLP, network_args=(16, 4, 0.2)):
        super().__init__()
        self.dim_notcond = dim_notcond
        self.dim_cond = dim_cond
        self.net_notcond = network(int(self.dim_notcond / 2), int(self.dim_notcond), *network_args)
        self.net_cond = network(self.dim_cond, int(self.dim_notcond), *network_args)

    def forward(self, x, x_condition):
        """
        Forward pass of the affine coupling layer.

        Args:
            x (torch.Tensor): Input tensor.
            x_condition (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor after applying the affine coupling layer.
            torch.Tensor: Log determinant of the Jacobian.
        """
        x.float()
        x_condition.float()
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = (self.net_notcond(x_b) * self.net_cond(x_condition)).chunk(2, dim=1)
        log_s = 0.636*1.9*torch.atan(log_s/1.9)
        s = torch.exp(log_s)
        # s = F.sigmoid(log_s)
        y_a = (s * x_a) + t
        y_b = x_b

        logdet = torch.sum(torch.log(s))

        return torch.cat([y_a, y_b], dim=1), logdet

    def backward(self, y, x_condition):
        """
        Backward pass of the affine coupling layer.

        Args:
            y (torch.Tensor): Input tensor.
            x_condition (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Log determinant of the Jacobian.
        """
        y.float()
        x_condition.float()
        y_a, y_b = y.chunk(2, dim=1)
        log_s, t = (self.net_notcond(y_b) * self.net_cond(x_condition)).chunk(2, dim=1)
        log_s = 0.636*1.9*torch.atan(log_s/1.9)
        s = torch.exp(log_s)
        # s = F.sigmoid(log_s)
        x_a = (y_a - t)/s 
        x_b = y_b

        logdet = torch.sum(torch.log(s))

        return torch.cat([x_a, x_b], dim=1), logdet
    
    
def find_bin(values, bin_boarders):
    #Make sure that a value=uppermost boarder is in last bin not last+1
    bin_boarders[..., -1] += 10**-6
    return torch.sum((values.unsqueeze(-1)>=bin_boarders),dim=-1)-1

#Takes a parametrisation of RQS and points and evaluates RQS or inverse
#Splines from [-B,B] identity else
#Bin widths normalized for 2B interval size
def eval_RQS(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
    """
    Function to evaluate points inside [-B,B] with RQS splines, given by the parameters.
    See https://arxiv.org/abs/1906.04032
    """
    #Get boarders of bins as cummulative sum
    #As they are normalized this goes up to 2B
    #Represents upper boarders
    bin_boardersx = torch.cumsum(RQS_bin_widths, dim=-1)

    #We shift so that they cover actual interval [-B,B]
    bin_boardersx -= RQS_B

    #Now make sure we include all boarders i.e. include lower boarder B
    #Also for bin determination make sure upper boarder is actually B and
    #doesn't suffer from rounding (order 10**-7, searching algorithm would anyways catch this)
    bin_boardersx = F.pad(bin_boardersx, (1,0), value=-RQS_B)
    bin_boardersx[...,-1] = RQS_B


    #Same for heights
    bin_boardersy = torch.cumsum(RQS_bin_heights, dim=-1)
    bin_boardersy -= RQS_B
    bin_boardersy = F.pad(bin_boardersy, (1,0), value=-RQS_B)
    bin_boardersy[...,-1] = RQS_B
    
    
    #Now with completed parametrisation (knots+derivatives) we can evaluate the splines
    #For this find bin positions first
    bin_nr = find_bin(X, bin_boardersy if inverse else bin_boardersx)
    
    #After we know bin number we need to get corresponding knot points and derivatives for each X
    x_knot_k = bin_boardersx.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    x_knot_kplus1 = bin_boardersx.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
    y_knot_k = bin_boardersy.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    y_knot_kplus1 = bin_boardersy.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
    delta_knot_k = RQS_knot_derivs.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    delta_knot_kplus1 = RQS_knot_derivs.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    

    #Evaluate splines, as shown in NSF paper
    s_knot_k = (y_knot_kplus1-y_knot_k)/(x_knot_kplus1-x_knot_k)
    if inverse:
        a = (y_knot_kplus1-y_knot_k)*(s_knot_k-delta_knot_k)+(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
        b = (y_knot_kplus1-y_knot_k)*delta_knot_k-(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
        c = -s_knot_k*(X-y_knot_k)
        
        Xi = 2*c/(-b-torch.sqrt(b**2-4*a*c))
        Y = Xi*(x_knot_kplus1-x_knot_k)+x_knot_k
        
        dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
        #No sum yet, so we can later keep track which X weren't in the intervall and need logdet 0
        logdet = -torch.log(dY_dX)

    else:
        Xi = (X-x_knot_k)/(x_knot_kplus1-x_knot_k)
        Y = y_knot_k+((y_knot_kplus1-y_knot_k)*(s_knot_k*Xi**2+delta_knot_k*Xi*(1-Xi)))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))
        dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
        logdet = torch.log(dY_dX)

    return Y, logdet


def RQS_global(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
    """Evaluates RQS spline, as given by parameters, inside [-B,B] and identity outside. Uses eval_RQS."""
    inside_interval = (X<=RQS_B) & (X>=-RQS_B)

    Y = torch.zeros_like(X)
    logdet = torch.zeros_like(X)
    
    Y[inside_interval], logdet[inside_interval] = eval_RQS(X[inside_interval], RQS_bin_widths[inside_interval,:], RQS_bin_heights[inside_interval,:], RQS_knot_derivs[inside_interval,:], RQS_B, inverse)
    Y[~inside_interval] = X[~inside_interval]
    logdet[~inside_interval] = 0
    
    #Now sum the logdet, zeros will be in the right places where e.g. all X components were 0
    logdet = torch.sum(logdet, dim=1)
    
    return Y, logdet


class NSF_CL(nn.Module):
    """Neural spline flow coupling layer. See https://arxiv.org/abs/1906.04032 for details.
    
    Implements the forward and backward transformation."""
    def __init__(self, dim_notcond, dim_cond, split=0.5, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
        """
        Constructs a Neural spline flow coupling layer.

        Parameters
        ----------

        dim_notcond : int
            The dimension of the input, i.e. the dimension of the data that will be transformed.
        dim_cond : int
            The dimension of the condition. If 0, the coupling layer is not conditioned.
        split : float, default: 0.5
            The fraction of the input that will be transformed. The rest will be left unchanged. The default is 0.5.
        K : int, default: 8
            The number of bins used for the spline.
        B : float, default: 3
            The interval size of the spline.
        network : nn.Module, default: MLP
            The neural network used to determine the parameters of the spline.
        network_args : tuple, default: (16,4,0.2)
            The arguments for the neural network.
        """
        super().__init__()
        self.dim = dim_notcond
        self.dim_cond = dim_cond
        self.K = K
        self.B = B
        
        self.split1 = int(self.dim*split)
        self.split2 = self.dim-self.split1
        
        self.net = network(self.split1, (3*self.K-1)*self.split2, *network_args)
        
        #Decide if conditioned or not
        if self.dim_cond>0:
            self.net_cond = network(dim_cond, (3*self.K-1)*self.split2, *network_args)
            
        
    def forward(self, x, x_cond):
        #Divide input into unchanged and transformed part
        unchanged, transform = x[..., :self.split1], x[..., self.split1:]

        #Get parameters from neural network based on unchanged part and condition
        if self.dim_cond>0:
            thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
        else:
            thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
        #Normalize NN outputs to get widths, heights and derivatives
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        #Evaluate splines
        transform, logdet = RQS_global(transform, widths, heights, derivs, self.B)

        return torch.hstack((unchanged,transform)), logdet
    
    def backward(self, x, x_cond):
        unchanged, transform = x[..., :self.split1], x[..., self.split1:]
        
        if self.dim_cond>0:
            thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
        else:
            thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        transform, logdet = RQS_global(transform, widths, heights, derivs, self.B, inverse=True)
        
        return torch.hstack((unchanged,transform)), logdet
    
    
class NSF_CL2(nn.Module):
    """Neural spline flow double coupling layer. First transforms the first half of the input, then the second half.
    Works only for even dimensions.
    
    Implements the forward and backward transformation."""
    def __init__(self, dim_notcond, dim_cond, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
        """
        Constructs a Neural spline flow double coupling layer.

        Parameters
        ----------

        dim_notcond : int
            The dimension of the input, i.e. the dimension of the data that will be transformed.
        dim_cond : int
            The dimension of the condition. If 0, the coupling layer is not conditioned.
        K : int, default: 8
            The number of bins used for the spline.
        B : float, default: 3
            The interval size of the spline.
        network : nn.Module, default: MLP
            The neural network used to determine the parameters of the spline.
        network_args : tuple, default: (16,4,0.2)
            The arguments for the neural network.
        """
        super().__init__()
        self.dim = dim_notcond
        self.dim_cond = dim_cond
        self.K = K
        self.B = B
        
        self.split = self.dim//2
        
        #Works only for even
        self.net1 = network(self.split, (3*self.K-1)*self.split, *network_args)
        self.net2 = network(self.split, (3*self.K-1)*self.split, *network_args)
        
        if dim_cond>0:
            self.net_cond1 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
            self.net_cond2 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
        
    def forward(self, x, x_cond):
        #Divide input into first and second half
        first, second = x[..., self.split:], x[..., :self.split]
        
        #Get parameters from neural network based on unchanged part and condition
        if self.dim_cond>0:
            thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
        #Normalize NN outputs to get widths, heights and derivatives
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        #Evaluate splines
        first, logdet = RQS_global(first, widths, heights, derivs, self.B)
        
        #Repeat for second half
        if self.dim_cond>0:
            thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        second, logdet_temp = RQS_global(second, widths, heights, derivs, self.B)
            
        logdet += logdet_temp
            
        return torch.hstack((second,first)), logdet
        
    def backward(self, x, x_cond):
        first, second = x[..., self.split:], x[..., :self.split]
        
        if self.dim_cond>0:
            thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
            
        second, logdet = RQS_global(second, widths, heights, derivs, self.B, inverse=True)
        
        if self.dim_cond>0:
            thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
            
        first, logdet_temp = RQS_global(first, widths, heights, derivs, self.B, inverse=True)
            
        logdet += logdet_temp
            
        return torch.hstack((second,first)), logdet


class NF_condGLOW(nn.Module):
    """
    Normalizing flow GLOW model with Affine coupling layers. Alternates coupling layers with GLOW convolutions Combines coupling layers and convolution layers.
    
    Args:
        n_layers (int): The number of flow layers. Flow layers consist of a coupling layer and a convolution layer.
        dim_notcond (int): The dimension of the input, i.e. the dimension of the data that will be transformed.
        dim_cond (int) : The dimension of the condition. If 0, the coupling layer is not conditioned.
        CL (nn.Module) : The coupling layer to use. Affine coupling layers is the only available for now
        **kwargs_CL (dict) : The arguments for the coupling layer
    
    Attributes:
        n_layers (int): The number of flow layers. Flow layers consist of a coupling layer and a convolution layer.
        dim_notcond (int): The dimension of the input, i.e. the dimension of the data that will be transformed.
        dim_cond (int) : The dimension of the condition. If 0, the coupling layer is not conditioned.
        prior(torch.distributions.MultivariateNormal): The prior distribution of the model.
        give_kwargs (dict): The arguments for the model.
    
    """

    def __init__(self, n_layers, dim_notcond, dim_cond, CL=AffineCoupling, **kwargs_CL):

        super().__init__()
        self.dim_notcond = dim_notcond
        self.dim_cond = dim_cond

        coupling_layers = [CL(dim_notcond, dim_cond, **kwargs_CL) for _ in range(n_layers)]
        conv_layers = [GLOW_conv(dim_notcond) for _ in range(n_layers)]


        self.layers = nn.ModuleList(itertools.chain(*zip(conv_layers,coupling_layers)))
        
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2), validate_args=False)

        #Information about hyperparameters accessible from outside
        #The function _get_right_hypers below will then reconstruct this back to __init__ arguments, if you change the model, change both.
        #This is needed in the background for recreating the same model in some cases like sampling with multiprocessing
        kwargs_parsed = kwargs_CL.copy()
        kwargs_parsed["network"] = "MLP"
        self.give_kwargs = {"n_layers":n_layers, "dim_notcond":dim_notcond, "dim_cond":dim_cond, "CL":"AffineCoupling", **kwargs_parsed}
        
    def forward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(x.device)
        
        for layer in self.layers:
            x, logdet_temp = layer.forward(x, x_cond)
            logdet += logdet_temp
            
        #Get p_z(f(x)) which is needed for loss function together with logdet
        prior_z_logprob = self.prior.log_prob(x).sum(-1)
        
        return x, logdet, prior_z_logprob
    
    def backward(self, y, x_cond):
        logdet = torch.zeros(y.shape[0]).to(y.device)
        
        for layer in reversed(self.layers):
            y, logdet_temp = layer.backward(y, x_cond)
            logdet += logdet_temp
            
        return y, logdet
    
    def sample(self, number, x_cond):
        """Samples from the prior and transforms the samples with the flow.
        
        Parameters
        ----------
        
        number : int
            The number of samples to draw. If a condition is given, the number of samples must be the same as the length of conditions.
        x_cond : torch.Tensor
            The condition for the samples. If dim_cond=0 enter torch.Tensor([]).

        """
        samples = self.backward( self.prior.sample(torch.Size((number,))), torch.from_numpy(x_cond).to(self.device) )[0]
        
        return samples
    
    def to(self, device):
        #Modified to also move the prior to the right device
        self.device = device
        super().to(device)
        self.prior = torch.distributions.Normal(torch.zeros(self.dim_notcond).to(device), torch.ones(self.dim_notcond).to(device))
        return self
    
    def get_pdf(self, x, x_cond):
        """Returns the probability density function of the flow at the given point.
        
        Parameters
        ----------
        
        x : torch.Tensor
            The point at which to evaluate the pdf.
        x_cond : torch.Tensor
            The condition for the point.
        """
        x = torch.from_numpy(x).to(self.device)
        x_cond = torch.from_numpy(x_cond).to(self.device)
        z, logdet, prior_z_logprob = self.forward(x, x_cond)
        pdf = torch.exp(prior_z_logprob + logdet)
        return pdf.cpu().detach().numpy()
    
    
    
def training_flow(flow:NF_condGLOW, data:pd.DataFrame, cond_names:list,  epochs, lr=2*10**-2, batch_size=1024, loss_saver=None, checkpoint_dir=None, gamma=0.998, optimizer_obj=None):
    
    writer = SummaryWriter()
    
    #Device the model is on
    device = flow.parameters().__next__().device

    data = data[data.columns.difference(['Galaxy_name'])]

    #Get index based masks for conditional variables
    mask_cond = np.isin(data.columns.to_list(), cond_names)
    mask_cond = torch.from_numpy(mask_cond).to(device)
    

    # Convert DataFrame to tensor (index based)
    data = torch.from_numpy(data.values).type(torch.float)
    # train_index, val_index = train_test_split(np.arange(data.shape[0]), test_size=0.2, random_state=42)
    train_set, val_set = train_test_split(data, test_size=0.2, random_state=42)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if optimizer_obj is None:
        optimizer = optim.Adam(flow.parameters(), lr=lr)
    else:
        optimizer = optimizer_obj

    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    #Save losses
    if loss_saver is None:
        losses = []
    else:
        losses = loss_saver

    #Total number of steps
    ct = 0

    start_time = time.perf_counter()

    best_loss = 1_000_000
    for e in tqdm(range(epochs)):
        running_loss = 0
        val_running_loss = 0
        for i, batch in enumerate(train_loader):
            x = batch.to(device)
            
            #Evaluate model
            z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob) 
            losses.append(loss.item())
            
            #Set gradients to zero
            optimizer.zero_grad()
            #Compute gradients
            loss.backward()
            #Update parameters
            optimizer.step()
            
            # Gather data and report
            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = e * len(train_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)    
                running_loss = 0.
            
            ct += 1
            #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
            if lr_schedule.get_last_lr()[0] <= 3*10**-6:
                decrease_step = 120
            else:
                decrease_step = 10

            #Update learning rate every decrease_step steps
            if ct % decrease_step == 0:
                lr_schedule.step()
                
        for i, batch in enumerate(val_loader): 
            x = batch.to(device)
            z, logdet, prior_z_logprob = flow(x[..., ~mask_cond], x[..., mask_cond])
            loss = -torch.mean(logdet+prior_z_logprob) 
            val_running_loss += loss.item()
            
        last_val_loss = val_running_loss / len(val_loader)
        writer.add_scalar('Loss/val', last_val_loss, e)
        if last_val_loss < best_loss:
            best_loss = last_val_loss
            torch.save(flow.state_dict(), f"{checkpoint_dir}checkpoint_best.pth")
            # print(f'state dict saved checkpoint_best')
            curr_time = time.perf_counter()
            np.save(f"{checkpoint_dir}losses_best.npy", np.array(best_loss))
        val_running_loss = 0.
            
            

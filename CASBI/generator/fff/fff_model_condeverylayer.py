import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from torch import nn

from CASBI.generator.fff.loss import  nll_surrogate
from CASBI.generator.fff.jacobian import compute_jacobian

class SkipConnection(torch.nn.Module):
    def __init__(self, module, cond_dim):
        super().__init__()
        self.module = module
        self.cond_dim = cond_dim

    def forward(self, x, cond):
        for name, layer in self.module.named_children():
            if isinstance(layer, torch.nn.Linear):
                x = torch.cat((x, cond), dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim, latent_dim, n_SC_layer):
        super().__init__()
        layers = [nn.Linear(dim + cond_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_SC_layer):
            layers.extend([nn.Linear(hidden_dim + cond_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim + cond_dim, latent_dim))
        self.layers = SkipConnection(nn.Sequential(*layers), cond_dim)

    def forward(self, x, cond):
        return self.layers(x, cond)

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, dim, n_SC_layer):
        super().__init__()
        layers = [nn.Linear(latent_dim + cond_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_SC_layer):
            layers.extend([nn.Linear(hidden_dim + cond_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim + cond_dim, dim))
        self.layers = SkipConnection(nn.Sequential(*layers), cond_dim)

    def forward(self, x, cond):
        return self.layers(x, cond)
    

class FreeFormFlow(torch.nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim, latent_dim, n_SC_layer, beta, device):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_SC_layer = n_SC_layer
        self.beta = beta
        self.device = device
        self.best_loss = 1000

        self.encoder = Encoder(dim, cond_dim, hidden_dim, latent_dim, n_SC_layer).to(device)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, dim, n_SC_layer).to(device)


        self.latent = torch.distributions.Independent(
            torch.distributions.Normal(loc=torch.zeros(latent_dim, device=self.device),
                                       scale=torch.ones(latent_dim, device=self.device), ),
            1)

    def train_model(self, n_epochs, batch_size, optimizer, train_set, val_set, snapshot_path='./snapshot/fff_snapshot/', runs_path='./runs/fff_runs/'):
        '''
        Train the FreeFormFlow model.

        Parameters
        ----------
        
        n_epochs (int): 
            The number of epochs to train for.
        batch_size (int):  
            The batch size for training.
        optimizer (torch.optim.Optimizer): 
            The optimizer for training.
        train_set (torch.utils.data.Dataset): 
            The training dataset.
        val_set (torch.utils.data.Dataset): 
            The validation dataset.
        snapshot_path (str, optional): 
            The path to save model snapshots. Defaults to './snapshot/fff_snpashot/'.
        runs_path (str, optional): 
            The path to save training logs. Defaults to './runs/fff_runs/'.
            
        Returns
        -------
        train_model: The trained FreeFormFlow model.
        '''
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        writer = SummaryWriter(log_dir=runs_path)

        for epoch in tqdm(range(n_epochs)):
            train_running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x, cond = batch[..., :2].float().to(self.device), batch[..., 2:].float().to(self.device)
                surrogate = nll_surrogate(x, cond, self.encoder, self.decoder)
                loss_reconstruction = ((x - surrogate.x1) ** 2).sum(-1).mean(-1)
                loss_nll = -self.latent.log_prob(surrogate.z) - surrogate.surrogate
                loss = self.beta*loss_reconstruction + loss_nll
                train_running_loss += loss.mean().item()
                loss.mean().backward()
                optimizer.step()
            writer.add_scalar('Loss/train', train_running_loss, epoch)

            val_running_loss = 0.0
            for batch in val_loader:
                self.eval()
                with torch.no_grad():
                    x, cond = batch[..., :2].float().to(self.device), batch[..., 2:].float().to(self.device)
                    x1, ll = self.log_prob(x, cond)
                    loss_nll = -ll.mean()
                    loss_reconstruction = ((x - x1) ** 2).sum(-1).mean(-1)
                    loss = self.beta*loss_reconstruction + loss_nll
                    val_running_loss += loss.mean().item()
                    if val_running_loss < self.best_loss:
                        self.best_loss = val_running_loss
                        save_path = os.path.join(snapshot_path, 'snapshot.pth')
                        torch.save(self.state_dict(), save_path)
                        print('Model saved at epoch', epoch, 'in file',  f'{save_path}')

            writer.add_scalar('Loss/val', val_running_loss, epoch)
            self.train()

    def log_prob(self, x, cond):
        '''
        Compute the log probability and the reconstructed x for the given input.

        Parameters
        ----------
        
        x (torch.Tensor): 
            The input data.
        cond (torch.Tensor): 
            The conditional data.

        Returns
        -------
        
        torch.Tensor: 
            The reconstructed x.
        torch.Tensor:  
            The log probability.

        '''
        z = self.encoder(x, cond)
        x1, jac_dec = compute_jacobian(z, cond, self.decoder)
        log_abs_jac_det = torch.linalg.slogdet(jac_dec).logabsdet
        log_prob = self.latent.log_prob(z) - log_abs_jac_det
        return x1, log_prob

    def sample(self, n_samples, cond):
        '''
        Generate samples from the FreeFormFlow model.

        Parameters
        ---------
        n_samples (int): 
            The number of samples to generate.
        cond (torch.Tensor): 
            The conditional data.

        Returns
        -------
        torch.Tensor: 
            The generated samples.

        '''
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        cond = cond.to(self.device)
        return self.decoder(z, cond)

            
        
        
        
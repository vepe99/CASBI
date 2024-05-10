import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from .loss import  nll_surrogate
from .jacobian import compute_jacobian

class SkipConnection(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, cond, *args, **kwargs):
        return x + self.inner(torch.cat((x, cond), dim=1), *args, **kwargs)
    
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
        
        encoder_SC_layers = [torch.nn.Linear(self.dim+self.cond_dim if i == 0 else self.hidden_dim, self.hidden_dim) if i % 2 == 0 else torch.nn.SiLU() for i in range(2*self.n_SC_layer - 1)]
        encoder_SC_layers.append(torch.nn.Linear(self.hidden_dim, self.latent_dim))
        
        
        decoder_SC_layers = [torch.nn.Linear(self.latent_dim+self.cond_dim if i == 0 else self.hidden_dim, self.hidden_dim) if i % 2 == 0 else torch.nn.SiLU() for i in range(2*self.n_SC_layer - 1)]
        decoder_SC_layers.append(torch.nn.Linear(self.hidden_dim, self.dim))

        self.encoder = SkipConnection(torch.nn.Sequential(*encoder_SC_layers)).to(self.device)
        self.decoder = SkipConnection(torch.nn.Sequential(*decoder_SC_layers)).to(self.device)
        
        self.latent = torch.distributions.Independent(
                        torch.distributions.Normal(loc=torch.zeros(latent_dim, device=self.device),
                                                    scale=torch.ones(latent_dim, device=self.device), ),  
                                                    1)
        
    def train_model(self, n_epochs, batch_size, optimizer, train_set, val_set, snapshot_path='./snapshot/fff_snpashot/', runs_path='./runs/fff_runs/'):
        
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
                        torch.save(self.state_dict(), os.path.join(runs_path, 'snapshot.pth'))
                        print('Model saved at epoch', epoch, 'in file',  f'{os.path.join(runs_path, "./snapshot.pth")}')
                    
            writer.add_scalar('Loss/val', val_running_loss, epoch)
            self.train()
                    
    def log_prob(self, x, cond):
        """
        return log_probability and the reconstructed x for the 
        """
        z = self.encoder(x, cond)
        x1, jac_dec = compute_jacobian(z, cond, self.decoder)
        log_abs_jac_det = torch.linalg.slogdet(jac_dec).logabsdet
        log_prob = self.latent.log_prob(z) - log_abs_jac_det
        return x1, log_prob
    
    def sample(self, n_samples, cond):
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        cond = cond.to(self.device)
        return self.decoder(z, cond)

            
        
        
        
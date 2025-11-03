import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import reduced plank constant
from scipy.constants import hbar

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1),   
            )
        
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        #x = x.view(x.size(0),-1)
        x = self.fully_connected(x)
        return x

def loss(x, model, n=0, lambda_p=0.0001, lambda_b=0.0001, N_p=500):
    w=1
    m=1

    E_n = torch.tensor((n + 0.5) * hbar * w).float().to(device)

    x = torch.from_numpy(x).float().to(device).requires_grad_(True)
    psi = model(x)
    dpsi = torch.autograd.grad(psi, # Outputs
                                x, # Inputs
                                grad_outputs=torch.ones_like(psi),
                                create_graph=True)[0]
    ddpsi = torch.autograd.grad(dpsi, # Outputs
                                x, # Inputs
                                grad_outputs=torch.ones_like(dpsi),
                                create_graph=True)[0]
    
    l_boundary = lambda_b/2 * (psi[0]**2 + psi[-1]**2)
    l_physics = ((-hbar/(2*m) * ddpsi + 0.5*m*w**2 * x**2 * psi - E_n * psi)**2).sum()
    l_physics *= lambda_p / N_p

    return l_boundary + l_physics


def train_PINN(model, optimizer, n=0, boundary=(-5,5), lambda_p=0.0001, lambda_b=0.0001, N_p=500, num_epochs = 1000):

    out_dict = {'train_loss': [],
                'val_loss': []}

    x_tr = np.linspace(boundary[0], boundary[1], N_p).reshape(-1, 1)

    train_loss = []
    val_loss = []
    
    for e in range(num_epochs):

        model.train()
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute Loss and backprop
        tr_loss = loss(x_tr, model,lambda_b=lambda_b,lambda_p=lambda_p,N_p=N_p)
        tr_loss.backward()
        # Update weight
        optimizer.step()      

        # Logging
        out_dict['train_loss'].append(tr_loss.item())  
        
        model.eval()
        with torch.no_grad():
            pass
        
    return out_dict


   
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # Init model
    pinn = PINN()

    # Send Model to GPU
    pinn.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(pinn.parameters(),lr=0.001)
    pinn.eval()
    with torch.no_grad():
        x_dummy = np.linspace(-5, 5, 10).reshape(-1,1)
        #print(pinn(torch.from_numpy(x_dummy).float().to(device)))
    
    train_PINN(pinn, optimizer)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import json
import time

from tqdm import tqdm

# Import reduced plank constant
#from scipy.constants import hbar
hbar = 1

import random
# Set seeds for reproducibility
SEED = 21
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Analytical Solution
def psi_analytical_n(x,n=0,m=1.0,w=1):
    y = np.sqrt(m*w/hbar)*x
    if n == 0:
        return np.exp(-0.5*y**2)
    if n == 1:
        return 2*y*np.exp(-0.5*y**2)
    if n == 2:
        return (4*y**2-2)*np.exp(-0.5*y**2)
    

class PINN(nn.Module):
    def __init__(self, act_fn = nn.Tanh()):
        super(PINN, self).__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(1,64),
            act_fn,
            #nn.BatchNorm1d(64),
            nn.Linear(64,64),
            act_fn,
            #nn.BatchNorm1d(64),
            nn.Linear(64,64),
            act_fn,
            #nn.BatchNorm1d(64),
            nn.Linear(64,64),
            act_fn,
            #nn.BatchNorm1d(64),
            nn.Linear(64,1),   
            )
        
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        #x = x.view(x.size(0),-1)
        x = self.fully_connected(x)
        return x

def loss(x, model, n=0, lambda_p=0.0001, lambda_b=0.0001, N_p=500,w=1,m=1):

    E_n = torch.tensor((n + 0.5) * hbar * w).float().to(device)

    #x = torch.from_numpy(x).float().to(device).requires_grad_(True)
    psi = model(x)
    dpsi = torch.autograd.grad(psi, # Outputs
                                x, # Inputs
                                grad_outputs=torch.ones_like(psi),
                                create_graph=True
                                )[0]
    ddpsi = torch.autograd.grad(dpsi, # Outputs
                                x, # Inputs
                                grad_outputs=torch.ones_like(dpsi),
                                create_graph=True)[0]
    
    l_boundary = lambda_b/2 * (psi[0]**2 + psi[-1]**2)
    l_physics = ((-hbar/(2*m) * ddpsi + 0.5*m*w**2 * x**2 * psi - E_n * psi)**2).sum()
    l_physics *= lambda_p / N_p

    return l_boundary + l_physics


def train_PINN(model, optimizer, n=0, boundary=(-5,5), lambda_p=0.0001, lambda_b=0.0001, N_p=500, N_val=500, num_epochs = 1000,m=1,w=1, log_every=100):

    model_dict = {
        'architecture': str(summary(model, input_size=(1,1))),
        'lr': 0.001,
        'n': n,
        'boundary': boundary,
        'lambda_p': lambda_p,
        'lambda_b': lambda_b,
        'N_p': N_p,
        'N_val': N_val,
        'num_epochs': num_epochs,
        'm': m,
        'w': w,
        'log_every': log_every,
        'loss_dict': {
            'train_loss': [],
            'val_loss': [],
            'x': []
        }
    }

    x_tr = np.linspace(boundary[0], boundary[1], N_p, endpoint=True).reshape(-1, 1)
    x_val = np.linspace(boundary[0], boundary[1], N_val, endpoint=True).reshape(-1, 1)
    model_dict['loss_dict']['x'] = x_val.tolist()
    x_tr = torch.from_numpy(x_tr).float().to(device).requires_grad_(True)
    x_val = torch.from_numpy(x_val).float().to(device).requires_grad_(True)
    
    for e in tqdm(range(num_epochs)):
        model.to(device)
        model.train()
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute Loss and backprop
        tr_loss = loss(x_tr, model,lambda_b=lambda_b,lambda_p=lambda_p,N_p=N_p,m=m,w=w,n=n)
        tr_loss.backward()
        # Update weight
        optimizer.step()      

        # Logging
        if e%log_every == 0 or e == num_epochs-1:
            print(f"Training loss at epoch {e}:",end=" ") 
            print(tr_loss.item())
        model_dict['loss_dict']['train_loss'].append(tr_loss.item())  
        
        model.eval()
        with torch.no_grad():
            psi = model(x_val).cpu().numpy()
            x = x_val.cpu().numpy()
            psi_analytical = psi_analytical_n(x,n=n,m=m,w=w)
            analytical_loss = psi_analytical - psi

            # Normalize
            #psi_analytical = np.linalg.norm(psi_analytical)
            #psi = np.linalg.norm(psi)
            psi_analytical /= np.linalg.norm(psi_analytical)
            psi /= np.linalg.norm(psi)

            #psi_analytical /= psi_analytical.sum()
            #psi /= psi.sum()

            if e%log_every == 0 or e == num_epochs-1:
                print(f"Analytical loss at epoch {e}:", end=" ") 
                print(analytical_loss.mean())
                plt.figure()
                plt.plot(x,psi_analytical,'r',label="Analytical Solution")
                plt.plot(x,psi, 'b', label="Prediction")
                plt.grid()
                plt.xlabel('Position (X)')
                plt.ylabel(r'$\Psi$')
                plt.legend()
                plt.title('Predictions Vs Analytical')
                plt.savefig(f"PredictionsVsAnalyticalAtEpoch{e}")
            model_dict['loss_dict']['val_loss'].append(float(analytical_loss.mean()))
        

    return model_dict

   
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
    optimizer = torch.optim.Adam(pinn.parameters(),lr=0.0001)
    
    epochs = 6000
    out = train_PINN(pinn, optimizer,lambda_p=1.0001, lambda_b=1.0001 ,N_p=10, N_val=500, num_epochs=epochs, log_every=epochs/4,n=2)

    # Good combos
    # epochs = 6000, N_p = 10
    
    epochs = range(epochs)
    train_loss = out['loss_dict']['train_loss']
    val_loss = out['loss_dict']['val_loss']
    plt.figure()
    plt.plot(epochs,train_loss,'b',label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(f"TrainLoss")


    plt.figure()
    plt.plot(epochs,val_loss, 'g', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(f"ValidationLoss")

    print(out['architecture'])

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f'settings_{timestamp}.json', 'w') as f:
        json.dump(out, f)

    """
    pinn.eval()
    with torch.no_grad():
        x_dummy = np.linspace(-5, 5, 10).reshape(-1,1)
        #print(pinn(torch.from_numpy(x_dummy).float().to(device)))
    """
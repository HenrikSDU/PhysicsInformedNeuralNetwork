import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import root_mean_squared_error

from tqdm import tqdm

# Import reduced plank constant
#from scipy.constants import hbar
hbar = 1

global integral_log
integral_log = 0

m = 1
w = 1


import random
# Set seeds for reproducibility
SEED = 20
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
    normalization_constant = np.power((m * w) / (np.pi * hbar), 1/4) * (1 / np.sqrt(np.pow(2, n) * math.factorial(n)))

    if n == 0:
        return normalization_constant * np.exp(-0.5*y**2)
    if n == 1:
        return normalization_constant * 2*y*np.exp(-0.5*y**2)
    if n == 2:
        return normalization_constant * (4*y**2-2)*np.exp(-0.5*y**2)
    

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


def loss(x, model, n=0, lambda_p=0.0001, lambda_p_end=0.0001, lambda_b=0.0001, lambda_b_end=0.0001, lambda_i=0.0001, lambda_i_end=0.0001, 
         adaptive_learning_rate=1, N_p=500,w=1,m=1,integral_resolution=1000, epochs=0, epoch_n=0):

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
    
    x_int_start = x[0].cpu().item()
    x_int_end = x[-1].cpu().item()
    x_int = np.linspace(x_int_start, x_int_end, integral_resolution, endpoint=True).reshape(-1, 1)
    x_int = torch.from_numpy(x_int).float().to(device)

    # i ran
    psi_x_int = (model(x_int))**2 # Get probability 
    #stepsize = (abs(x_int_end) + abs(x_int_start)) / integral_resolution
    #psi_integral = stepsize * psi_x_int.sum()

    stepsize = (x_int_end - x_int_start) / (integral_resolution - 1)
    
    # Use trapezoidal rule for better accuracy
    psi_integral = stepsize * (
        0.5 * psi_x_int[0] + 
        psi_x_int[1:-1].sum() + 
        0.5 * psi_x_int[-1]
    )

    global integral_log
    integral_log += 1
    if integral_log % 1000 == 0:
        print(f"Integral:", end='')
        print(psi_integral)

    # Loss coef interpolator 3000
    lambda_i_inter = ((lambda_i_end - lambda_i) / epochs) * epoch_n + lambda_i
    lambda_p_inter = ((lambda_p_end - lambda_p) / epochs) * epoch_n + lambda_p
    lambda_b_inter = ((lambda_b_end - lambda_b) / epochs) * epoch_n + lambda_b

    
    l_integral = lambda_i_inter * (1 - psi_integral)**2

    l_boundary = lambda_b_inter/2 * (psi[0]**2 + psi[-1]**2)
    l_physics = ((-hbar/(2*m) * ddpsi + 0.5*m*w**2 * x**2 * psi - E_n * psi)**2).sum()
    l_physics *= lambda_p_inter / N_p

    return l_boundary + l_physics + l_integral, {'L_b':l_boundary.item(),'L_p':l_physics.item(),'L_i':l_integral.item()}


def train_PINN(model, optimizer, n=0, boundary=(-5,5), lambda_p=0.0001, lambda_p_end=0.0001, lambda_b=0.0001, lambda_b_end=0.0001, lambda_i=0.0001, lambda_i_end=0.0001, 
               adaptive_learning_rate=1, N_p=500, N_val=500, num_epochs=1000,m=1,w=1, log_every=100,integral_resolution=1000):

    model_dict = {
        'architecture': str(summary(model, input_size=(1,1))),
        'lr': 0.001,
        'n': n,
        'boundary': boundary,
        'lambda_p': lambda_p,
        'lambda_b': lambda_b,
        'lambda_i': lambda_i,
        'lambda_b_end': lambda_b_end,
        'lambda_p_end': lambda_p_end,
        'N_p': N_p,
        'N_val': N_val,
        'integral_resolution': integral_resolution,
        'num_epochs': num_epochs,
        'm': m,
        'w': w,
        'log_every': log_every,
        'loss_dict': {
            'train_loss': [], # Sum of losses
            'boundary_loss': [],
            'physics_loss': [],
            'integral_loss': [],
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
        tr_loss, L_dict = loss(x_tr, model,
                                lambda_b=lambda_b,
                                lambda_b_end=lambda_b_end,
                                lambda_p=lambda_p,
                                lambda_p_end=lambda_p_end,
                                lambda_i=lambda_i,
                                lambda_i_end=lambda_i_end,
                                adaptive_learning_rate=adaptive_learning_rate,
                                N_p=N_p,m=m,w=w,n=n,
                                integral_resolution=integral_resolution,
                                epochs=num_epochs,
                                epoch_n=e)
        tr_loss.backward()
        # Update weight
        optimizer.step()      

        # Logging
        if e%log_every == 0 or e == num_epochs-1:
            print(f"Training loss at epoch {e}:",end=" ") 
            print(tr_loss.item())
        
        model_dict['loss_dict']['train_loss'].append(tr_loss.item())  
        model_dict['loss_dict']['boundary_loss'].append(L_dict['L_b'])
        model_dict['loss_dict']['physics_loss'].append(L_dict['L_p'])
        model_dict['loss_dict']['integral_loss'].append(L_dict['L_i'])
        
        model.eval()
        with torch.no_grad():
            psi = model(x_val).cpu().numpy()
            x = x_val.cpu().numpy()
            psi_analytical = psi_analytical_n(x,n=n,m=m,w=w)
            #analytical_loss = nn.MSELoss(psi, psi_analytical) 
            #analytical_loss = psi_analytical - psi
            analytical_loss = np.sqrt(((psi_analytical - psi) ** 2).mean())
            #analytical_loss = root_mean_squared_error(psi_analytical, psi)

            # Normalize
            #psi_analytical = np.linalg.norm(psi_analytical)
            #psi = np.linalg.norm(psi)
            #psi_analytical /= np.linalg.norm(psi_analytical)
            #psi /= np.linalg.norm(psi)

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
                plt.title(f'Predictions Vs Analytical at Epoch: {e}')
                plt.savefig(f"test_run_images/PredictionsVsAnalyticalAtEpoch{e}")
                print("Figure Saved!")
            model_dict['loss_dict']['val_loss'].append(float(analytical_loss.mean()))
        

    return model_dict

   
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("The code will run on CUDA.")
        device = torch.device('cuda')
    elif torch.mps.is_available():
        print("The code will run on MPS.")
        device = torch.device('mps')
    else:
        print("The code will run on CPU")
        device = torch.device('cpu')

    #################################################
    #                    Training                   #
    #################################################
    # Init model
    pinn = PINN()

    # Send Model to GPU
    pinn.to(device)


    # Initialize optimizer
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.0001, weight_decay=0.0001)
    
    epochs = 9000
    out = train_PINN(pinn, optimizer,
                     lambda_p=50.0, 
                     lambda_p_end=100.0, 
                     lambda_b=5.0,
                     lambda_b_end=5.0,
                     lambda_i=5.0,
                     lambda_i_end=2.5,
                     adaptive_learning_rate=1,
                     integral_resolution=1000,
                     N_p=100, 
                     N_val=500, 
                     num_epochs=epochs, 
                     log_every=epochs/18,n=1)

    # Good combos
    # epochs = 6000, N_p = 10

    #################################################
    #                    Plotting                   #
    #################################################
    epochs = range(epochs)
    train_loss = out['loss_dict']['train_loss']
    integral_loss = out['loss_dict']['integral_loss']
    physics_loss = out['loss_dict']['physics_loss']
    boundary_loss = out['loss_dict']['boundary_loss']
    plt.figure()
    plt.plot(epochs,train_loss,'b',label="Train Loss")
    plt.plot(epochs,integral_loss,'c',label="Integral Loss")
    plt.plot(epochs,boundary_loss,'r',label="Boundary Loss")
    plt.plot(epochs,physics_loss,'limegreen',label="Physics Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.title('Train Loss Over Epochs')
    plt.savefig(f"test_run_images/TrainLoss")


    plt.figure()
    val_loss = out['loss_dict']['val_loss']
    plt.plot(epochs,val_loss, 'limegreen', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.title('Validation Loss Over Epochs')
    plt.savefig(f"test_run_images/ValidationLoss")


    # Example: assuming your losses are lists/arrays in out['loss_dict']
    epochs = range(len(out['loss_dict']['train_loss']))
    integral_loss = np.array(out['loss_dict']['integral_loss'])
    boundary_loss = np.array(out['loss_dict']['boundary_loss'])
    physics_loss = np.array(out['loss_dict']['physics_loss'])

    # Compute total and shares
    total_loss = integral_loss + boundary_loss + physics_loss
    integral_share = integral_loss / total_loss
    boundary_share = boundary_loss / total_loss
    physics_share = physics_loss / total_loss

    # Plot shares as line plots
    plt.figure(figsize=(8,5))
    plt.plot(epochs, integral_share, 'c', label="Integral Share")
    plt.plot(epochs, boundary_share, 'r', label="Boundary Share")
    plt.plot(epochs, physics_share, 'limegreen', label="Physics Share")
    plt.xlabel('Epoch')
    plt.ylabel('Loss Share (fraction of total)')
    plt.title('Relative Share of Loss Components Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_run_images/LossShares.png")
    #plt.show()

    # Optional: stacked area chart for clearer composition
    plt.figure(figsize=(8,5))
    plt.stackplot(epochs,
                integral_share, boundary_share, physics_share,
                labels=['Integral Share','Boundary Share','Physics Share'],
                colors=['c','r','limegreen'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss Share (stacked)')
    plt.title('Stacked Shares of Loss Components')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("test_run_images/LossShares_Stacked.png")
    #plt.show()


    #################################################
    #                    Logging                    #
    #################################################

    print(out['architecture'])

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f'settings_{timestamp}.json', 'w') as f:
        json.dump(out, f)

    
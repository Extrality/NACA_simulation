import yaml
import numpy as np
import metrics
from simulation_generator import simulation

# Global path where the airFoil2DInit folder is and where the simulations are gonna be done.
glob_path = 'Simulations/'

with open('params.yaml', 'r') as f: # hyperparameters of the model
    params = yaml.safe_load(f)

# Properties of air at 1.01325hPa
NU = -3.400747e-6 + 3.452139e-8*params['temperature'] + 1.00881778e-10*params['temperature']**2 - 1.363528e-14*params['temperature']**3

n = 5 # Size of the dataset is 2n
Reynolds = np.random.uniform(2, 6, 2*n)*1e6
AoA = np.random.uniform(-5, 15, 2*n)

# 4-digits
M = np.random.uniform(0, 7, n)
P_4 = np.random.uniform(0, 7, n)
P_4[P_4 < 1.5] = 0
XX_4 = np.random.uniform(5, 20, n)

# 5-digits
L = np.random.uniform(0, 4, n)
P_5 = np.random.uniform(3, 8, n)
Q = np.random.randint(2, size = n)
XX_5 = np.random.uniform(5, 20, n)

design_space = []
for i in range(len(M)):
    design_space.append([Reynolds[i], AoA[i], M[i], P_4[i], XX_4[i]])
for i in range(len(L)):
    design_space.append([Reynolds[i + len(M)], AoA[i + len(M)], L[i], P_5[i], Q[i], XX_5[i]])

index = np.arange(len(Reynolds))
np.random.shuffle(index)
design_space_shuffle = []
for i in range(len(design_space)):
    design_space_shuffle.append(design_space[index[i]])

for sim in design_space_shuffle:    
    params['Uinf'] = np.round(sim[0]*NU, 3) 
    params['aoa'] = np.round(sim[1], 3) # Angle of attack. (< 15)
    params['digits'] = tuple(np.round(sim[2:], 3)) # 4 or 5-digits for the naca airfoil.
    if np.abs(params['aoa']) <= 10:
        params['n_iter'] = 20000
    else:
        params['n_iter'] = 40000

    init_path = glob_path + 'airFoil2DInit/' # OpenFoam initial case path (don't forget the '/' at the end of the path)

    digits = ''
    for digit in params['digits']:
        digits = digits + '_' + str(digit) 
    path = glob_path + 'airFoil2D_' + params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + digits + '/' 
    
    simulation(init_path, path, params, figure = True, compute_grad = True, VTK = True)
    res = metrics.plot_residuals(path, params)
    coef = metrics.plot_coef_convergence(path, params)
    with open(glob_path + 'manifest', 'a') as manifest:
        manifest.write('airFoil2D_' + params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + digits + '\n')
import yaml, argparse
import numpy as np
import metrics
from simulation_generator import simulation

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--init', help = 'Only generate the mesh (default: 0)', default = 0, type = int)
parser.add_argument('-g', '--gradient', help = 'Compute the term of the RANS equations as a post-processing (default: 0)', default = 0, type = int)
parser.add_argument('-v', '--vtk', help = 'Generate the VTK files from the simulation (default: 1)', default = 1, type = int)
parser.add_argument('-f', '--figure', help = 'Save an image of the airfoil in the simulation folder (default: 1)', default = 1, type = int)
args = parser.parse_args()

# Global path where the airFoil2DInit folder is and where the simulations are gonna be done.
glob_path = 'Simulations/'

with open('params.yaml', 'r') as f: # hyperparameters of the model
    params = yaml.safe_load(f)

# Properties of air at 1.01325hPa
NU = -3.400747e-6 + 3.452139e-8*params['temperature'] + 1.00881778e-10*params['temperature']**2 - 1.363528e-14*params['temperature']**3
  
params['Uinf'] = np.round(params['reynolds']*NU, 3) 
params['aoa'] = np.round(params['aoa'], 3) # Angle of attack. (< 15)
params['digits'] = tuple(np.round(params['digits'], 3)) # 4 or 5-digits for the naca airfoil.

init_path = glob_path + 'airFoil2DInit/' # OpenFoam initial case path (don't forget the '/' at the end of the path)

digits = ''
for digit in params['digits']:
    digits = digits + '_' + str(digit) 
path = glob_path + 'airFoil2D_' + params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + digits + '/' 

simulation(init_path, path, params, just_init = bool(args.init), figure = bool(args.figure), compute_grad = bool(args.gradient), VTK = bool(args.vtk))
res = metrics.plot_residuals(path, params)
coef = metrics.plot_coef_convergence(path, params)
# NACA_simulation
In this repository, you will find different python scripts to run incompressible or compressible Reynolds-Averaged-Simulations over NACA airfoils.

# Requirements
- OpenFOAM v2112 (You can install it by following those [instructions](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian) for Ubuntu)
- Python 3.9.12
- NumPy 1.23.1
- PyYAML 6.0
- Seaborn 0.11.2

# Simulation
To launch a simulation, enter your parameters in the ```params.yaml``` file and run:
```
python main.py -i 0 -g 1 -v 1 -f 1
```

# Usage
```usage: main.py [-h] [-i INIT] [-g GRADIENT] [-v VTK] [-f FIGURE]

optional arguments:
  -h, --help            show this help message and exit
  -i INIT, --init INIT  Only generate the mesh (default: 0)
  -g GRADIENT, --gradient GRADIENT
                        Compute the term of the RANS equations as a post-processing (default: 0)
  -v VTK, --vtk VTK     Generate the VTK files from the simulation (default: 1)
  -f FIGURE, --figure FIGURE
                        Save an image of the airfoil in the simulation folder (default: 1)
```

# Dataset
Those scripts have been used to generate the AirfRANS dataset proposed at the NeurIPS 2022 Datasets and Benchmarks Track conference. You can find the paper submission [here](https://openreview.net/forum?id=Zp8YmiQ_bDC&referrer=%5Bthe%20profile%20of%20Florent%20Bonnet%5D(%2Fprofile%3Fid%3D~Florent_Bonnet1)). In particular, the script ```dataset_generator.py``` run multiple simulations by sampling Reynolds number and Angle of Attack as explained in the associated paper.

This script can be re-used to run multiple new random simulations.

# Mesh parameters
The mesh is generated with the blockMesh utility available in the OpenFOAM suite. The block definition is given in the following ![scheme](https://github.com/Extrality/NACA_simulation/blob/main/mesh_scheme.pdf?raw=true).
Some of the parameters contained in the ```params.yaml``` file are for the mesh generation. Parameters are defined as:
- ```L```: Size of the domain in meters
- ```y_h```: Heigth of the first cell of the boundary layer 
- ```y_hd```: Heigth of the furthest first cell of the trail (at vertex 1 in the scheme)
- ```x_h```: Width of the smallest cell at the leading edge (at vertex 8 in the scheme)
- ```y_exp```: Expansion ration in the y-direction
- ```x_exp```: Expansion ration in the x-direction on the airfoil (edge between vertices 8 and 11)

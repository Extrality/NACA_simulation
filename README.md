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
python main.py
```

# Dataset
Those scripts have been used to generate the AirfRANS dataset proposed at the NeurIPS 2022 Datasets and Benchmarks Track conference. You can find the paper submission [here](https://openreview.net/forum?id=Zp8YmiQ_bDC&referrer=%5Bthe%20profile%20of%20Florent%20Bonnet%5D(%2Fprofile%3Fid%3D~Florent_Bonnet1)). In particular, the script ```dataset_generator.py``` run multiple simulations by sampling Reynolds number and Angle of Attack as explained in the associated paper.

This script can be re-used to run multiple new random simulations.

# Mesh parameters

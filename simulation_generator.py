import numpy as np
import os, shutil, glob
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from naca_generator import naca_generator

sns.set()

# Properties of air at sea level and 293.15K 
NU = 1.56e-5
C = 343.2

def angle_to_origin(a, b, alpha):
    c = b - a
    # Compute the middle point on the segment [a, b]
    mid = (a + b)/2

    # Compute the distance between a and b
    d1 = np.linalg.norm(c)

    # Compute the distance between the middle point and the center of the circle
    d = d1/(2*np.tan(alpha/2))

    # Compute the angle between (b - mid) and the x-axis
    theta = np.arctan(c[1]/c[0])

    # Rotate the center of the circle
    center = np.array([0, d])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = rot@center + mid

    return center
    
def coef_grading(a, b, c, type = 'L, h, e'):
    if type == 'L, h, e':
        L, h, expansion = a, b, c
        N = int(np.log(1 - (1 - expansion)*L/h)/np.log(expansion))
        delta = expansion**(N - 1)
        H = h*delta
        results = (N, H, delta)
    
    elif type == 'L, h, delta':
        L, h, delta = a, b, c
        expansion = 1 + h*(delta - 1)/(L - h*delta)
        N = int(np.log(1 - (1 - expansion)*L/h)/np.log(expansion))
        H = h*delta
        results = (N, H, expansion)
    
    elif type == 'L, N, e':
        L, N, expansion = a, b, c
        delta = expansion**(N - 1)
        if expansion < 1:
            H = L*(1 - expansion)/(1 - expansion**N)
            h = L*(1 - 1/expansion)/(1 - (1/expansion)**N)
        else:
            H = L*(1 - 1/expansion)/(1 - (1/expansion)**N)
            h = L*(1 - expansion)/(1 - expansion**N)
        results = (H, h, delta)
    
    elif type == 'L, N, h':
        L, N, h = a, b, c
        f = lambda x: (h*(x**N) - L*x + L - h)
        df = lambda x: (h*N*(x**(N - 1)) - L)
        if (N*h > L):
            old_expansion = 0.5
        else:
            old_expansion = 2
        cond = True

        while cond:            
            new_expansion = old_expansion - f(old_expansion)/df(old_expansion)
            cond = (np.abs(new_expansion - old_expansion) > 1e-15)
            old_expansion = new_expansion

        expansion = old_expansion
        delta = expansion**(N - 1)
        H = h*delta
        results = (H, expansion, delta)
        
    return results

def dict_grading(L, y_h, y_hd, x_h, y_exp, x_exp, x_expd, aoa, geometry):
    arg_upper = geometry[:, 1].argmax()
    arg_lead = geometry[:, 0].argmin()    
    x_upper = geometry[arg_upper, 0]
    y_upper = geometry[arg_upper, 1]
    x_lead = geometry[arg_lead, 0]
    y_lead = geometry[arg_lead, 1]
    y_lower = geometry[np.abs(geometry[:, 0] - x_upper) < 1e-2, 1].min()
    arg_lower = np.where(geometry[:, 1] - y_lower == 0)[0][0]
    x_lower = geometry[arg_lower, 0]    

    d_ULead = np.sqrt(((geometry[arg_upper:arg_lead] - geometry[arg_upper + 1: arg_lead + 1])**2).sum(axis = 1)).sum()
    d_DLead = np.sqrt(((geometry[arg_lead:arg_lower] - geometry[arg_lead + 1:arg_lower + 1])**2).sum(axis = 1)).sum()

    d_UTrail = np.sqrt(((geometry[:arg_upper] - geometry[1:arg_upper + 1])**2).sum(axis = 1)).sum()    
    d_DTrail = np.sqrt(((geometry[arg_lower:-1] - geometry[arg_lower + 1:])**2).sum(axis = 1)).sum()

    alpha = np.pi*20/180
    x_mid = L/6
    a, b = np.array([1, 0]), np.array([x_mid, L])
    center = angle_to_origin(a, b, alpha)

    x_angle = (L - 1)*np.sin(aoa*np.pi/180)

    # yGrading
    N_y, _, delta_y = coef_grading(L, y_h, y_exp)
    L_U, L_D = L - x_angle, L + x_angle
    _, _, delta_yu = coef_grading(L_U, N_y, y_hd, type = 'L, N, h')
    _, _, delta_yd = coef_grading(L_D, N_y, y_hd, type = 'L, N, h')

    # leadGrading
    N_Ulead, H_Ulead, delta_Ulead = coef_grading(d_ULead, x_h, x_exp)
    delta_Ulead = 1/delta_Ulead
    N_Dlead, H_Dlead, delta_Dlead = coef_grading(d_DLead, x_h, x_exp)
    delta_Dlead = 1/delta_Dlead

    # xMAeroGrading
    expansion = 1.001*(1 - H_Ulead/d_UTrail)
    N_UM, h_UM_aero, delta_UM_aero = coef_grading(d_UTrail, H_Ulead, expansion)
    delta_UM_aero = 1/delta_UM_aero

    delta_DM_aero = H_Dlead/h_UM_aero
    N_DM, _, _ = coef_grading(d_DTrail, h_UM_aero, delta_DM_aero, type = 'L, h, delta')

    # expansion = 1.001*(1 - H_Dlead/d_DTrail)
    # N_DM, h_DM_aero, delta_DM_aero = coef_grading(d_DTrail, H_Dlead, expansion)
    # delta_DM_aero = 1/delta_DM_aero

    # xUGrading
    L_U = np.pi/2*L
    h_UU = (x_mid - x_upper)/N_UM    
    _, _, delta_UU = coef_grading(L_U, N_Ulead, h_UU, type = 'L, N, h')

    h_DU = (x_mid - x_upper)/N_DM    
    _, _, delta_DU = coef_grading(L_U, N_Dlead, h_DU, type = 'L, N, h')

    # xMGrading
    delta_M = 1
    h_UM = delta_M*h_UU
    h_DM = delta_M*h_DU

    # xDTrailGrading
    N_D, _, delta_D_trail = coef_grading(L - 1, h_UM_aero, x_expd)
    delta_D_trail = 1/delta_D_trail

    # N_DD, _, delta_DD_trail = coef_grading(L - 1, h_UM_aero, x_expd)
    # delta_DD_trail = 1/delta_DD_trail

    # xDGrading
    _, _, delta_UD = coef_grading(L - x_mid, N_D, h_UM, type = 'L, N, h')
    delta_UD = 1/delta_UD

    _, _, delta_DD = coef_grading(L - x_mid, N_D, h_DM, type = 'L, N, h')
    delta_DD = 1/delta_DD

    results = dict()
    results['xMin'] = -L
    results['xMid'] = x_mid
    results['xAngle'] = x_angle

    results['yCells'] = N_y
    results['xUUCells'] = N_Ulead
    results['xDUCells'] = N_Dlead
    results['xUMCells'] = N_UM
    results['xDMCells'] = N_DM
    results['xDCells'] = N_D
    # results['xDDCells'] = N_DD

    results['yGrading'] = delta_y
    results['yUGrading'] = delta_yu
    results['yDGrading'] = delta_yd
    results['xUUGrading'] = delta_UU
    results['xDUGrading'] = delta_DU
    results['xUMAeroGrading'] = delta_UM_aero
    results['xDMAeroGrading'] = delta_DM_aero
    results['xMGrading'] = delta_M
    results['xDTrailGrading'] = delta_D_trail
    # results['xDDTrailGrading'] = delta_DD_trail
    results['xUDGrading'] = delta_UD
    results['xDDGrading'] = delta_DD
    results['leadUGrading'] = delta_Ulead
    results['leadDGrading'] = delta_Dlead

    results['xOrigin'] = center[0]
    results['yOrigin'] = center[1]

    results['argUpper'] = arg_upper
    results['argLead'] = arg_lead
    results['argLower'] = arg_lower
    results['xLead'] = x_lead
    results['yLead'] = y_lead
    results['xUpper'] = x_upper
    results['yUpper'] = y_upper
    results['xLower'] = x_lower
    results['yLower'] = y_lower

    return results

def blockMeshDict_generator(path, geometry, params):
    with open(path + 'system/blockMeshDict.orig', 'r') as file:
        line_list = file.read().splitlines()

    # Domain boundaries
    line_list[23] = '\txMin\t' + str(params['xMin']) + ';'
    line_list[27] = '\txMid\t' + str(params['xMid']) + ';'
    line_list[28] = '\txAngle\t' + str(params['xAngle']) + ';'

    # Cells number
    line_list[31] = '\tyCells\t' + str(params['yCells']) + ';'
    line_list[32] = '\txUUCells\t' + str(params['xUUCells']) + ';'
    line_list[33] = '\txDUCells\t' + str(params['xDUCells']) + ';'
    line_list[34] = '\txUMCells\t' + str(params['xUMCells']) + ';'
    line_list[35] = '\txDMCells\t' + str(params['xDMCells']) + ';'
    line_list[36] = '\txDCells\t' + str(params['xDCells']) + ';'

    # Grading
    line_list[39] = '\tyGrading\t' + str(params['yGrading']) + ';'
    line_list[40] = '\tyUGrading\t' + str(params['yUGrading']) + ';'
    line_list[41] = '\tyDGrading\t' + str(params['yDGrading']) + ';'
    line_list[42] = '\txUUGrading\t' + str(params['xUUGrading']) + ';'
    line_list[43] = '\txDUGrading\t' + str(params['xDUGrading']) + ';'
    line_list[44] = '\txUMAeroGrading\t' + str(params['xUMAeroGrading']) + ';'
    line_list[45] = '\txDMAeroGrading\t' + str(params['xDMAeroGrading']) + ';'
    line_list[47] = '\txDTrailGrading\t' + str(params['xDTrailGrading']) + ';'
    line_list[48] = '\txUDGrading\t' + str(params['xUDGrading']) + ';'
    line_list[49] = '\txDDGrading\t' + str(params['xDDGrading']) + ';'
    line_list[50] = '\tleadUGrading\t' + str(params['leadUGrading']) + ';'
    line_list[51] = '\tleadDGrading\t' + str(params['leadDGrading']) + ';'

    # Mid-arc
    line_list[54] = '\txOrigin\t' + str(params['xOrigin']) + ';'
    line_list[55] = '\tyOrigin\t' + str(params['yOrigin']) + ';'

    # Aerofoil
    line_list[60] = '\txLead\t' + str(params['xLead']) + ';'
    line_list[61] = '\tyLead\t' + str(params['yLead']) + ';'
    line_list[64] = '\txUpper\t' + str(params['xUpper']) + ';'
    line_list[65] = '\tyUpper\t' + str(params['yUpper']) + ';'
    line_list[66] = '\txLower\t' + str(params['xLower']) + ';'
    line_list[67] = '\tyLower\t' + str(params['yLower']) + ';'

    end = line_list[169:]
    line_list = line_list[:169]

    line_list.extend(['', '\tspline 10 11', '\t('])
    for pt in geometry[:params['argUpper'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 0)')

    line_list.extend(['\t)', '', '\tspline 22 23', '\t('])
    for pt in geometry[:params['argUpper'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 1)')

    line_list.extend(['\t)', '', '\tspline 11 8', '\t('])
    for pt in geometry[params['argUpper']:params['argLead'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 0)')
    
    line_list.extend(['\t)', '', '\tspline 23 20', '\t('])
    for pt in geometry[params['argUpper']:params['argLead'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 1)')
    
    line_list.extend(['\t)', '', '\tspline 8 9', '\t('])
    for pt in geometry[params['argLead']:params['argLower'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 0)')

    line_list.extend(['\t)', '', '\tspline 20 21', '\t('])
    for pt in geometry[params['argLead']:params['argLower'] + 1]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 1)')

    line_list.extend(['\t)', '', '\tspline 9 10', '\t('])
    for pt in geometry[params['argLower']:]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 0)')

    line_list.extend(['\t)', '', '\tspline 21 22', '\t('])
    for pt in geometry[params['argLower']:]:
        line_list.append('\t\t(' + str(pt[0]) + ' ' + str(pt[1]) + ' 1)')
    line_list.append('\t)')

    line_list.extend(end)

    with open(path + 'system/blockMeshDict', 'w') as file:
        for line in line_list:
            file.write(line + '\n')

def system_generator(path, u_inf, aoa, n_proc, n_iter = 10000, compressible = False):
    a = np.array([np.cos(aoa*np.pi/180), np.sin(aoa*np.pi/180)])

    with open(path + 'system/controlDict.orig', 'r') as file:
        line_list = file.read().splitlines()

    line_list[15] = 'Uinf\t' + str(u_inf) + ';'
    line_list[25] = 'endTime\t' + str(n_iter) + ';'
    line_list[105] = '\t    liftDir\t     (' + str(-a[1]) + ' ' + str(a[0]) + ' 0);'
    line_list[108] = '\t    dragDir\t     (' + str(a[0]) + ' ' + str(a[1]) + ' 0);'        

    if compressible:
        line_list[17] = 'application     rhoSimpleFoam;' # Change solver
        line_list[63] = '	    rho\trho;' # Add density for integral forces computation
        line_list[85] = '	    rho\trho;' # Add density for force coeffs computation
        line_list[91] = '\t    pRef            1.013e5;' # Absolute pressure for compressible simulation
        line_list[162] = '            mut'
        line_list[163] = '            muEff'
        line_list[164] = '            devRhoReff'

        file = open(path + 'system/fvSolution', 'r')
        line_list_fv = file.read().splitlines()

        # Come nack to SIMPLE algorithm for stability
        line_list_fv[41] = '    consistent		 no;'
        line_list_fv[59] = '        p               0.5;'
        line_list_fv[64] = '        e               0.7;'
        line_list_fv[65] = '        U               0.7;'
        line_list_fv[66] = '        nuTilda         0.7;'
        line_list_fv[67] = '        k               0.7;'
        line_list_fv[68] = '        omega           0.7;'

        with open(path + 'system/fvSolution', 'w') as file:
            for line in line_list_fv:
                file.write(line + '\n')
    else:
        os.remove(path + 'system/fvOptions')

    with open(path + 'system/controlDict', 'w') as file:
        for line in line_list:
            file.write(line + '\n')
    
    with open(path + 'system/decomposeParDict', 'r') as file:
        line_list = file.read().splitlines()

    line_list[16] = 'numberOfSubdomains ' + str(n_proc) + ';'

    with open(path + 'system/decomposeParDict', 'w') as file:
        for line in line_list:
            file.write(line + '\n')

def init_generator(path, u_inf, aoa, L, turbulence, y_1 = None, compressible = False):
    a = np.array([u_inf*np.cos(aoa*np.pi/180), u_inf*np.sin(aoa*np.pi/180)])
    shutil.copytree(path + '0.orig/', path + '0')

    with open(path + '0/U', 'r') as file:
        line_list = file.read().splitlines()

    line_list[18] = 'field\t\t(' + str(a[0]) + ' ' + str(a[1]) + ' 0);'

    with open(path + '0/U', 'w') as file:
        for line in line_list:
            file.write(line + '\n')

    if turbulence == 'SA':
        with open(path + 'constant/turbulenceProperties', 'r') as file:
            line_list_turb = file.read().splitlines()

        line_list_turb[20] = '\tRASModel\tSpalartAllmaras;'
        os.remove(path + '0/k')
        os.remove(path + '0/omega')
    
    elif turbulence == 'SST':
        if isinstance(y_1, float):
            Re_L = u_inf*L/NU
            k = 1e-3*u_inf**2/Re_L
            omega = 5*u_inf/L
            omega_wall = 6*NU/0.075/y_1**2

            with open(path + 'constant/turbulenceProperties', 'r') as file:
                line_list_turb = file.read().splitlines()

            line_list_turb[20] = '\tRASModel\tkOmegaSST;'

            with open(path + '0/k', 'r') as file:
                line_list = file.read().splitlines()
            line_list[18] = 'field\t\t' + str(k) + ';'
            with open(path + '0/k', 'w') as file:
                for line in line_list:
                    file.write(line +'\n')

            with open(path + '0/omega', 'r') as file:
                line_list = file.read().splitlines()
            line_list[18] = 'field\t\t' + str(omega) + ';'
            line_list[27] = '\t\tvalue\t\tuniform ' + str(omega_wall) + ';'
            with open(path + '0/omega', 'w') as file:
                for line in line_list:
                    file.write(line +'\n')
            
            os.remove(path + '0/nuTilda')
        
        else:
            raise ValueError('The height of the first aerofoil cell "y_1" must be given as a float number.')    

    with open(path + 'constant/turbulenceProperties', 'w') as file:
        for line in line_list_turb:
            file.write(line +'\n')
    
    if compressible:
        with open(path + '0/p', 'r') as file:
            line_list_comp = file.read().splitlines()

        line_list_comp[16] = 'field           1.013e5;'
        line_list_comp[18] = 'dimensions      [1 -1 -2 0 0 0 0];' # Change the dimension to the real pressure dimension.

        with open(path + '0/p', 'w') as file:
            for line in line_list_comp:
                file.write(line + '\n')
        
        os.remove(path + 'constant/transportProperties')
    
    else:
        os.remove(path + 'constant/thermophysicalProperties')
        os.remove(path + '0/alphat')
        os.remove(path + '0/T')

def simulation(init_path, path, params, just_init = False, figure = False, compute_grad = False, VTK = False):
    '''
    Run a Reynolds-Averaged-Simulation over airfoils on OpenFOAM v2112.

    Args:
        init_path (string): Path where the initial conditions and simulation dictionnaries are given.
        path (string): Path where the simulation is going to be saved.
        params (dict): Parameters of the simulation.
        just_init (bool, optional): If ``True``, only do the mesh generation without the simulation. Default: ``False``
        figure (bool, optional): If ``True``, save an image of the airfoil in the simulation folder. Default: ``False``
        compute_grad (bool, optional): If ``True``, compute the different terms of the RANS equations as a post-processing. Default: ``False``
        VTK (bool, optional): If ``True``, generate VTK files from the simulation. Default: ``False``
    '''
    Re, Ma = params['Uinf']/NU, params['Uinf']/C
    print('Reynolds number: {0:.3}\
        \nMach number: {1:.3}\
        \nUinf: {2:.3}\
        \nAoA: {3:.3}\
        \nNACA:'.format(Re, Ma, float(params['Uinf']), float(params['aoa'])), str(params['digits']),
        '\nTurbulence model: ' + params['turbulence'] + '\nCompressible: ' + str(params['compressible']) + '\n')

    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.copytree(init_path, path)

    print('Generating NACA airfoil.')
    geometry = naca_generator(params['digits'], nb_samples = 2000, scale = 1, origin = (0, 0), cosine_spacing = True, verbose = False, CTE = True)
    if figure:
        fig, ax = plt.subplots(figsize = (15, 15*params['digits'][-1]/100))
        ax.scatter(geometry[:, 0], geometry[:, 1])
        ax.set_title('NACA ' + str(params['digits']))
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        fig.savefig(path + 'naca_' + str(params['digits']) + '.png', bbox_inches = 'tight', dpi = 150);

    print('Generating simulation dictionnaries.')
    coef_grad = dict_grading(params['L'], params['y_h'], params['y_hd'], params['x_h'], params['y_exp'], params['x_exp'], params['x_expd'], params['aoa'], geometry)
    blockMeshDict_generator(path, geometry, coef_grad)
    system_generator(path, params['Uinf'], params['aoa'], params['n_proc'], n_iter = params['n_iter'], compressible = params['compressible'])
    init_generator(path, params['Uinf'], params['aoa'], 2*params['L'], params['turbulence'], params['y_h'], params['compressible'])    

    wd = os.getcwd()
    os.chdir(path)
    print('Generating mesh.')
    subprocess.run('blockMesh > log.blockMesh', shell = True)

    print('Checking mesh.')
    subprocess.run('checkMesh > log.checkMesh', shell = True)
    open(params['turbulence'] + '_' + str(params['Uinf']) + '_' + str(params['aoa']) + '_' + str(params['digits']) + '.foam', 'w')
    if just_init:
        print('\nInitialization done!')
    
    else:
        print('Simulation running.')
        if params['compressible']:
            if params['n_proc'] == 1:
                subprocess.run('rhoSimpleFoam > log.rhoSimpleFoam', shell = True)
            else:
                subprocess.run('decomposePar > log.decomposePar', shell = True)
                # subprocess.run('mpirun -np ' + str(params['n_proc']) + ' renumberMesh -parallel -overwrite > log.renumberMesh', shell = True)
                subprocess.run('mpirun -np ' + str(params['n_proc']) + ' rhoSimpleFoam -parallel > log.rhoSimpleFoam', shell = True)
            subprocess.run('foamLog log.rhoSimpleFoam > log.foamLog', shell = True)
        else:
            if params['n_proc'] == 1:
                subprocess.run('simpleFoam > log.simpleFoam', shell = True)
            else:
                subprocess.run('decomposePar > log.decomposePar', shell = True)
                # subprocess.run('mpirun -np ' + str(params['n_proc']) + ' renumberMesh -parallel -overwrite > log.renumberMesh', shell = True)
                subprocess.run('mpirun -np ' + str(params['n_proc']) + ' simpleFoam -parallel > log.simpleFoam', shell = True)
            subprocess.run('foamLog log.simpleFoam > log.foamLog', shell = True)        
        subprocess.run('reconstructPar > log.reconstructPar', shell = True)        
        for path_dir in glob.glob('processor*'):
            shutil.rmtree(path_dir)
        if compute_grad:
            print('Computing gradient.')
            subprocess.run('postProcess -func "components(U)" > log', shell = True)
            subprocess.run('postProcess -func "grad(Ux)" > log', shell = True)
            subprocess.run('postProcess -func "grad(Uy)" > log', shell = True)
            subprocess.run('postProcess -func "div(grad(Ux))" > log', shell = True)
            subprocess.run('postProcess -func "div(grad(Uy))" > log', shell = True)
            subprocess.run('postProcess -func "grad(p)" > log', shell = True)
            os.remove('log')
            for file in ['Ux.gz', 'Uy.gz', 'Uz.gz']:
                os.remove(str(params['n_iter']) + '/' + file)
        if VTK:
            print('Generating VTK.')
            if params['compressible']:
                subprocess.run("foamToVTK -noZero -fields '(p U T rho nut wallShearStress Ma)' > log.foamToVTK", shell = True)
            else:
                subprocess.run("foamToVTK -noZero -fields '(p U nut wallShearStress)' > log.foamToVTK", shell = True)
        print('\nSimulation done!')

    os.chdir(wd)
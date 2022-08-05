import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Properties of air at sea level and 298.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5

def surface_coefficients(airfoil, params, sorted = True):
    qInf = 0.5*params['Uinf']**2
    # if params['compressible']:  # OpenFOAM case path (don't forget the '/' at the end of the path)
    #     fold = 'airFoil2D_rho_' + params['turbulence'] + '_' + str(params['digits']) + '_' + str(params['aoa']) + '_' + str(params['Uinf']) + '_' + str(params['n_iter']) + '/'
    # else:
    #     fold = 'airFoil2D_' + params['turbulence'] + '_' + str(params['digits']) + '_' + str(params['aoa']) + '_' + str(params['Uinf']) + '_' + str(params['n_iter']) + '/'
    # aerofoil = pv.read(path + 'VTK/' + fold + 'boundary/aerofoil.vtp').cell_centers()
    aerofoil = airfoil.cell_centers()

    if sorted:
        jump = np.argwhere(np.abs(aerofoil.points[:-1, 0] - aerofoil.points[1:, 0]) > 5e-2) + 1
        n_extrado = len(aerofoil.points) - jump[2, 0] + jump[0, 0]
        points = np.vstack([
                aerofoil.points[jump[2, 0]:, 0], aerofoil.points[:jump[0, 0], 0], 
                aerofoil.points[jump[0, 0]:jump[1, 0], 0][::-1], aerofoil.points[jump[1, 0]:jump[2, 0], 0][::-1]
            ])
        pressure = np.vstack([
                aerofoil.cell_data['p'][jump[2, 0]:], aerofoil.cell_data['p'][:jump[0, 0]], 
                aerofoil.cell_data['p'][jump[0, 0]:jump[1, 0]][::-1], aerofoil.cell_data['p'][jump[1, 0]:jump[2, 0]][::-1]
            ])

        wss = np.vstack([
                aerofoil.cell_data['wallShearStress'][jump[2, 0]:, :2], aerofoil.cell_data['wallShearStress'][:jump[0, 0], :2], 
                aerofoil.cell_data['wallShearStress'][jump[0, 0]:jump[1, 0], :2][::-1], aerofoil.cell_data['wallShearStress'][jump[1, 0]:jump[2, 0], :2][::-1]
            ])
        wss = np.linalg.norm(wss, axis = 1)
    else:
        points = aerofoil.points[:, 0]
        pressure = aerofoil.cell_data['p']
        wss = np.linalg.norm(aerofoil.cell_data['wallShearStress'][:, :2], axis = 1)

    if params['compressible']:
        c_p = np.vstack([points, (pressure - P_ref)/qInf/RHO]).T
    else:
        c_p = np.vstack([points, pressure/qInf]).T
    c_l = np.vstack([points, wss/qInf/RHO]).T

    if sorted:
        results = (c_p, c_l, n_extrado)
    else:
        results = (c_p, c_l)
    return results

def compare_surface_coefs(coefs1, coefs2, extrado = True, path = None):
    ycp1, ycp2, c_p1, c_p2 = coefs1[0][:, 0], coefs2[0][:, 0], coefs1[0][:, 1], coefs2[0][:, 1]
    ycl1, ycl2, c_f1, c_f2 = coefs1[1][:, 0], coefs2[1][:, 0], coefs1[1][:, 1], coefs2[1][:, 1]

    fig, ax = plt.subplots(2, figsize = (20, 10))    
    if extrado:
        n_extrado1, n_extrado2 = coefs1[2], coefs2[2]
        ax[0].scatter(ycp1[:n_extrado1], c_p1[:n_extrado1], label = 'Extrado 1')
        ax[0].scatter(ycp1[:n_extrado1], c_p1[:n_extrado1], color = 'r', marker = 'x', label = 'Intrado 1')
        ax[0].scatter(ycp2[:n_extrado2], c_p2[:n_extrado2], color = 'y', label = 'Extrado 2')
        ax[0].scatter(ycp2[:n_extrado2], c_p2[:n_extrado2], color = 'r', marker = 'x', label = 'Intrado 2')

        ax[1].scatter(ycl1[:n_extrado1], c_f1[:n_extrado1], label = 'Extrado 1')
        ax[1].scatter(ycl1[:n_extrado1], c_f1[:n_extrado1], color = 'r', marker = 'x', label = 'Intrado 1')
        ax[1].scatter(ycl2[:n_extrado2], c_f2[:n_extrado2], color = 'y', label = 'Extrado 2')
        ax[1].scatter(ycl2[:n_extrado2], c_f2[:n_extrado2], color = 'g', marker = 'x', label = 'Intrado 2')

    else:
        ax[0].scatter(ycp1, c_p1, label = 'Experiment 1')
        ax[0].scatter(ycp2, c_p2, color = 'y', label = 'Experiment 2')

        ax[1].scatter(ycl1, c_f1, label = 'Experiment 1')
        ax[1].scatter(ycl2, c_f2, color  = 'y', label = 'Experiment 2')
    
    ax[0].invert_yaxis()
    ax[0].set_xlabel('x/c')
    ax[1].set_xlabel('x/c')
    ax[0].set_ylabel(r'$C_p$')
    ax[1].set_ylabel(r'$C_f$')
    ax[0].set_title('Pressure coefficient')
    ax[1].set_title('Skin friction coefficient')
    ax[0].legend(loc = 'best')
    ax[1].legend(loc = 'best')

    if path != None:
        fig.savefig(path + 'surface_coefs.png', bbox_inches = 'tight', dpi = 150)
    
def boundary_layer(airfoil, internal, x, params, y = .1, resolution = int(1e4), rotation = True):
    u_inf = params['Uinf']
    # if params['compressible']:  # OpenFOAM case path (don't forget the '/' at the end of the path)
    #     fold = 'airFoil2D_rho_' + params['turbulence'] + '_' + str(params['digits']) + '_' + str(params['aoa']) + '_' + str(params['Uinf']) + '_' + str(params['n_iter']) + '/'
    # else:
    #     fold = 'airFoil2D_' + params['turbulence'] + '_' + str(params['digits']) + '_' + str(params['aoa']) + '_' + str(params['Uinf']) + '_' + str(params['n_iter']) + '/'
    # aerofoil = pv.read(path + 'VTK/' + fold + 'boundary/aerofoil.vtp')
    aerofoil = airfoil.compute_normals(point_normals = False, inplace = True, flip_normals = True).cell_centers()

    jump = np.argwhere(np.abs(aerofoil.points[:-1, 0] - aerofoil.points[1:, 0]) > 5e-2) + 1
    n_extrado = len(aerofoil.points) - jump[2, 0] + jump[0, 0]
    points = np.vstack([
            aerofoil.points[jump[2, 0]:], aerofoil.points[:jump[0, 0]], 
            aerofoil.points[jump[0, 0]:jump[1, 0]][::-1], aerofoil.points[jump[1, 0]:jump[2, 0]][::-1]
        ])
    normals = np.vstack([
            aerofoil.cell_data['Normals'][jump[2, 0]:], aerofoil.cell_data['Normals'][:jump[0, 0]], 
            aerofoil.cell_data['Normals'][jump[0, 0]:jump[1, 0]][::-1], aerofoil.cell_data['Normals'][jump[1, 0]:jump[2, 0]][::-1]
        ])

    arg = np.argmin(np.abs(points[:n_extrado, 0] - x))
    a, b = points[arg], points[arg] + y*normals[arg]

    # internal = pv.read(path + 'VTK/' + fold + 'internal.vtu')
    bl = internal.sample_over_line(a, b, resolution = resolution)
    
    if rotation:
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        u = (bl.point_data['U']*(rot@normals[arg])).sum(axis = 1)
        v = (bl.point_data['U']*normals[arg]).sum(axis = 1)
    else:
        u = bl.point_data['U'][:, 0]
        v = bl.point_data['U'][:, 1]
    
    nut = bl.point_data['nut']
    yc = bl.points[:, 1] - a[1]

    return yc, u/u_inf, v/u_inf, nut/NU

def compare_boundary_layer(coefs1, coefs2, ylim = .1, path = None):
    yc1, u1, v1, nut1 = coefs1
    yc2, u2, v2, nut2 = coefs2

    fig, ax = plt.subplots(1, 3, figsize = (30, 10))
    ax[0].scatter(u1, yc1, label = 'Experiment 1')
    ax[0].scatter(u2, yc2, label = 'Experiment 2', color = 'r', marker = 'x')
    ax[0].set_xlabel(r'$u/U_\infty$')
    ax[0].set_ylabel(r'$(y-y_0)/c$')
    # ax[0].set_xlim([-0.2, 1.4])
    ax[0].set_ylim([0, ylim])
    ax[0].legend(loc = 'best')

    ax[1].scatter(v1, yc1, label = 'Experiment 1')
    ax[1].scatter(v2, yc2, label = 'Experiment 2', color = 'r', marker = 'x')
    ax[1].set_xlabel(r'$v/U_\infty$')
    ax[1].set_ylabel(r'$(y-y_0)/c$')
    # ax[1].set_xlim([-0.2, 0.2])
    ax[1].set_ylim([0, ylim])
    ax[1].legend(loc = 'best')

    ax[2].scatter(nut1, yc1, label = 'Experience 1')
    ax[2].scatter(nut2, yc2, label = 'Experience 2', color = 'r', marker = 'x')
    ax[2].set_ylim([0, ylim])
    ax[2].set_xlabel(r'$\nu_t/\nu$')
    ax[2].set_ylabel(r'$(y-y_0)/c$')
    ax[2].legend(loc = 'best')

    if path != None:
        fig.savefig(path + 'boundary_layer.png', bbox_inches = 'tight', dpi = 150)

def plot_residuals(path, params):
    datas = dict()
    if params['turbulence'] == 'SA':
        fields = ['Ux', 'Uy', 'p', 'nuTilda']
    elif params['turbulence'] == 'SST':
        fields = ['Ux', 'Uy', 'p', 'k', 'omega']
    for field in fields:
        data = np.loadtxt(path + 'logs/' + field +'_0')[:, 1]
        datas[field] = data

    if params['turbulence'] == 'SA':
        fig, ax = plt.subplots(2, 2, figsize = (20, 20))
        ax[1, 1].plot(datas['nuTilda'])
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('nuTilda residual')
        ax[1, 1].set_xlabel('Number of iterations')

    elif params['turbulence'] == 'SST':
        fig, ax = plt.subplots(3, 2, figsize = (30, 20))
        ax[1, 1].plot(datas['k'])
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('k residual')
        ax[1, 1].set_xlabel('Number of iterations')

        ax[2, 0].plot(datas['omega'])
        ax[2, 0].set_yscale('log')
        ax[2, 0].set_title('omega residual')
        ax[2, 0].set_xlabel('Number of iterations');
    
    ax[0, 0].plot(datas['Ux'])
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_title('Ux residual')

    ax[0, 1].plot(datas['Uy'])
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_title('Uy residual')

    ax[1, 0].plot(datas['p'])
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title('p residual')
    ax[1, 0].set_xlabel('Number of iterations');

    fig.savefig(path + 'residuals.png', bbox_inches = 'tight', dpi = 150)

    return datas

def plot_coef_convergence(path, params):
    datas = dict()
    datas['c_d'] = np.loadtxt(path + 'postProcessing/forceCoeffs1/0/coefficient.dat')[:, 1]
    datas['c_l'] = np.loadtxt(path + 'postProcessing/forceCoeffs1/0/coefficient.dat')[:, 3]
    c_d, c_l = datas['c_d'][-1], datas['c_l'][-1]

    fig, ax = plt.subplots(2, figsize = (30, 15))
    ax[0].plot(datas['c_d'])
    ax[0].set_ylim([.5*c_d, 1.5*c_d])
    ax[0].set_title('Drag coefficient')
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel(r'$C_D$')

    ax[1].plot(datas['c_l'])
    ax[1].set_title('Lift coefficient')
    ax[1].set_ylim([.5*c_l, 1.5*c_l])
    ax[1].set_ylabel(r'$C_L$')
    ax[1].set_xlabel('Number of iterations');

    print('Drag coefficient: {0:.5}, lift coefficient: {1:.5}'.format(c_d, c_l))

    fig.savefig(path + 'coef_convergence.png', bbox_inches = 'tight', dpi = 150)

    return datas, c_d, c_l
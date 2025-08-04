import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d

def interpolate_space_axis(data, target_points=60):
    
    T, S = data.shape
    x_old = np.linspace(0, S - 1, S)
    x_new = np.linspace(0, S - 1, target_points)
    data_interp = np.zeros((T, target_points))
    for t in range(T):
        f = interp1d(x_old, data[t, :], kind='linear')
        data_interp[t, :] = f(x_new)
    return x_new, data_interp

def plot_corrf_3d_better(corr, corr_pred, savename, space_interp_points=60):
   
    
    time_len = corr.shape[0]
    time_axis = np.arange(time_len)

    
    x_new, corr_smooth = interpolate_space_axis(corr, space_interp_points)
    _, pred_smooth = interpolate_space_axis(corr_pred, space_interp_points)

    
    X,Y = np.meshgrid(x_new, time_axis)
    #X, Y = np.meshgrid(x_new, time_axis)

    
    fig = plt.figure(figsize=(10, 5), dpi=300)
    -
    mask = Y > 11
    pred_smooth_masked = np.where(mask, np.nan, pred_smooth)
    corr_smooth_masked = np.where(mask, np.nan, corr_smooth)
    #z_min = min(np.nanmin(pred_smooth_masked), np.nanmin(corr_smooth_masked))
    #z_max = max(np.nanmin(pred_smooth_masked), np.nanmin(corr_smooth_masked))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, pred_smooth_masked, cmap=cm.Blues, edgecolor='none', antialiased=True)
    #ax1.set_title('Forecast', fontsize=12)
    ax1.set_xlabel(r'Space interval $(*\Delta x)$', fontsize=10)
    ax1.set_ylabel(r'Time interval $(*\Delta t)$', fontsize=10)
    ax1.set_zlabel('R', fontsize=10)
    ax1.set_zlim([0.8, 1.05])
    ax1.set_ylim([0,11])
    ax1.text(x=X.min(), y=Y.min(), z=1.045, s='(a)', fontsize=12, ha='left', va='top')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
   

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, corr_smooth_masked, cmap=cm.Oranges, edgecolor='none', antialiased=True)
    #ax2.set_title('Experimental', fontsize=12)
    ax2.set_xlabel(r'Space interval $(*\Delta x)$', fontsize=10)
    ax2.set_ylabel(r'Time interval $(*\Delta t)$', fontsize=10)
    ax2.set_zlabel('R', fontsize=10)
    ax2.set_zlim([0.8, 1.05])
    ax2.set_ylim([0,11])
    ax2.text(X.min(), Y.min(), 1.045, s='(b)', fontsize=12, ha='left', va='top')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    ax1.w_xaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax1.w_yaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax1.w_zaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax2.w_xaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax2.w_yaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax2.w_zaxis.set_pane_color((0.97, 0.97, 0.97, 0.6))
    ax1.view_init(elev=20, azim=70)
    ax2.view_init(elev=20, azim=70)


    for ax in [ax1, ax2]:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]['color'] = (0.6, 0.6, 0.6, 0.3)
            axis._axinfo["grid"]['linestyle'] = '--'

    plt.tight_layout()
    plt.savefig(savename, bbox_inches='tight')
    #plt.show()

import numpy as np 
pred = np.load('pred_output_50.npy')
target = np.load('target_output_50.npy')
# Calculate the mean squared error
from plot_results_funcs import plot_mean_std_cp, plot_fluctuation_power_spectrum, plot_cp, calculate_corrf, plot_corrf, plot_cd, plot_cl, plot_corrf_compare, plot_corrf_3d

corr_pred = calculate_corrf(nx=7, nt=700, Cp=pred[:, 13:21])
corr_target = calculate_corrf(nx=7, nt=700, Cp=target[:, 13:21])
plot_corrf_3d_better(corr_target, corr_pred, 'corr_3d_surface_new_3.pdf')

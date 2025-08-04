# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import matplotlib.ticker as ticker

def plot_mean_std_cp(pred, target, pred_low):
    """
    Plot the mean and standard deviation of Cp values for each pressure tap,
    comparing the forecast (with and without Beta) and experimental data.
    

    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        pred_low (ndarray): shape (T, N), (No Beta)
        save_path (str): 'mean_std_plot.pdf'
    """
    
    mean_pred = np.mean(pred, axis=0)
    std_pred = np.std(pred, axis=0)
    mean_target = np.mean(target, axis=0)
    std_target = np.std(target, axis=0)
    mean_pred_low = np.mean(pred_low, axis=0)
    std_pred_low = np.std(pred_low, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    x = np.arange(1, 27)  # 空间点编号

    # ax.errorbar(x, mean_pred, yerr=std_pred, fmt='o-', label='Forecast (with Beta)', capsize=3)
    # ax.errorbar(x, mean_target, yerr=std_target, fmt='s--', label='Experimental data', capsize=3)
    # ax.errorbar(x, mean_pred_low, yerr=std_pred_low, fmt='^:', label='Forecast (No Beta)', capsize=3)
    # Forecast (with Beta)：
    eb1 = ax.errorbar(x, mean_pred, yerr=std_pred,
    fmt='o-', label=r'Forecast (with $\beta$)',
    capsize=3, elinewidth=2  
        )
    # Experimental data：
    eb2 = ax.errorbar(
    x, mean_target, yerr=std_target,
    fmt='s--', label='Experimental data',
    capsize=3 
    )

    # Forecast (No Beta)
    eb3 = ax.errorbar(
    x, mean_pred_low, yerr=std_pred_low,
    fmt='^:', label=r'Forecast (without $\beta$)',
    capsize=3   
    )
    
    for ebar in [eb2, eb3]:
        for line in ebar[2]:  
            line.set_linestyle('--')
            line.set_linewidth(0.5)


    ax.set_xlabel('Pressure tap No.', fontsize=16)
    ax.set_ylabel(r'$\overline{C}_p \pm \sigma $',  fontsize=16)

    
    ax.legend(prop={'size': 16})

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim([0.5, 26.5])  # set the x-axis limits

    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = 'mean_std_plot.pdf'  
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_fluctuation_power_spectrum(pred, target, pred_low, point_No, fs, nperseg):
    """
    plot the **fluctuation power spectrum** for a specified pressure tap,
    comparing the forecast (with and without Beta) and experimental data.
    
    
    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        point_No (int): the index of the pressure tap to plot
        fs (float): sampling frequency, 400 Hz
        nperseg (int): Welch length, 256
        pred_low (ndarray): shape (T, N), (No Beta)
        save_path (str): 'fluctuation_spectrum_point_{point_No}.pdf'
    """
    
    pred_fluc = pred[:, point_No] - np.mean(pred[:, point_No])
    target_fluc = target[:, point_No] - np.mean(target[:, point_No])
    pred_low_fluc = pred_low[:, point_No] - np.mean(pred_low[:, point_No])

  
    f_pred, Pxx_pred = welch(pred_fluc, fs=fs, nperseg=nperseg)
    f_pred_plot = f_pred*0.003
    f_target, Pxx_target = welch(target_fluc, fs=fs, nperseg=nperseg)
    f_target_plot = f_target*0.003
    f_pred_low, Pxx_pred_low = welch(pred_low_fluc, fs=fs, nperseg=nperseg)
    f_pred_low_plot = f_pred_low*0.003

    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    ax.semilogy(f_pred_plot, Pxx_pred, label=r'Forecast (with $\beta$)', color='blue', linewidth=1.5)
    ax.semilogy(f_target_plot, Pxx_target, label='Experimental data', color='orange', linewidth=1.5)
    ax.semilogy(f_pred_low_plot, Pxx_pred_low, label=r'Forecast (without $\beta$)', color='green', linewidth=1, linestyle='--')
    ax.axvline(x=0.33, color='black', linestyle='-.', linewidth=1)
    ylim = ax.get_ylim()
    ax.text(0.331, ylim[1]*0.099, r'($f_t^\prime$)', color='black', fontsize=16)
    ax.text(0.179, ylim[1]*0.099,'Frequency cutoff ', color='black', fontsize=16)

    ax.set_xlabel(r'$St$', fontsize=16)
    ax.set_ylabel('Fluctuating Power Spectrum Density', fontsize=16)
    ax.set_xlim([0, 0.5])

    
    ax.legend(prop={'size': 16})
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontname('Times New Roman')
        label.set_fontsize(15)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = f'fluctuation_spectrum_point_{point_No}.pdf'
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_cp(pred, target, point_No):
    """
    plot the Cp values for a specified pressure tap,
    comparing the forecast and experimental data.
    

    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        save_path (str):'cp_plot.pdf'
    """
    
    fig, ax= plt.subplots(figsize=(4, 3), dpi=600)
    
    x = np.arange(pred.shape[0])  

    
    ax.plot(x, pred[:, point_No], linewidth=1, linestyle='-', label='Forecast')

    
    ax.plot(x, target[:, point_No], linewidth=0.5, linestyle='-.', label='Experimental data')
    ax.set_xlim([0, 1000])

    
    ax.set_xlabel(r'$\Delta T / \delta t$', fontsize=16)
    ax.set_ylabel(r'$C_p$', fontsize=16)
    
    
    ax.legend(prop={'size': 14})

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # plt.title('Mean Cp at Each Spatial Point')  
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_path = f'cp_plot_point_{point_No}.pdf'  
    
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()  

def plot_power_spectrum(pred, target, point_No, fs, nperseg):
    """
    plot the power spectral density for a specified pressure tap,
    comparing the forecast and experimental data.
    

    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        point_No (int)
        save_path (str): 'power_spectrum_plot.pdf'
    """
    from scipy.signal import welch

    
    f_pred, Pxx_pred = welch(pred[:, point_No], fs=fs, nperseg=nperseg)
    f_target, Pxx_target = welch(target[:, point_No], fs=fs, nperseg=nperseg)

    
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    
    ax.semilogy(f_pred, Pxx_pred, label='Forecast', color='blue')
    ax.semilogy(f_target, Pxx_target, label='Experimental data', color='orange')

    ax.set_xlabel('Frequency [Hz]', fontsize=16)
    ax.set_ylabel('Power Spectral Density [dB/Hz]', fontsize=16)

    
    ax.legend(prop={'size': 14})

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = f'power_spectrum_plot_point_{point_No}.pdf'  
    fig.savefig(save_path, format='pdf',  bbox_inches='tight')
    plt.close() 

    
def calculate_corrf(nx, nt, Cp):
    corr=np.zeros([nt,nx])
    for dx in range(nx):#dx is the spatial dimension
        for n in range(nt): # n is used to determine the temporal dimension
            data1=Cp[0:len(Cp)-1-n,0:7-dx]
            data2=Cp[n:len(Cp)-1, dx:7]
            prod_cross=np.multiply(data1,data2)
            prod1=np.multiply(data1,data1)
            prod2=np.multiply(data2,data2)
            corr[n][dx] = prod_cross.mean(axis=(0,1))/np.sqrt(prod1.mean(axis=(0,1))*prod2.mean(axis=(0,1)))
    import pickle
    with open('corr.pkl', 'wb') as f:
        pickle.dump(corr, f)
    return corr

def plot_corrf(corr, savename):
    #import pickle
    #with open(corr_path, 'rb') as f:
    #    corr = pickle.load(f)
        
    import matplotlib.pyplot as plt
    #plt.rcParams['font.family'] = 'Times New Roman'
   
    X = np.arange(corr.shape[1]) #spatial points
    Y = np.arange(corr.shape[0]) # temporal
    X, Y = np.meshgrid(X, Y)
    save_x = []
    save_y = []


    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 20), sharex=True, dpi=1200)
    x_slices = np.arange(corr.shape[1])


    for i, ax in enumerate(axes):
        x_slice = x_slices[i]
        z_slice = corr[:, x_slice]  
        y_slice = Y[:, x_slice]  

   
        ax.plot(y_slice, z_slice, color='blue', linewidth=1.5, marker='o', markersize=3)

    
        ax.set_title('Spatial interval {:}dx'.format(x_slice+1), fontsize=16)
        ax.set_ylabel('R', fontsize=16)
        ax.tick_params(direction='in')
    
        ax.set_xlim(0, 22)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        x_index = np.argmax(z_slice[0:7])
        max_value = np.max(z_slice[0:7])
        y_value = y_slice[x_index]
        save_x.append(max_value)
        save_y.append(y_value)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    


    axes[-1].set_xlabel(r'$Time scale (*\delta t)$', fontsize=16)



    plt.tight_layout()


    # plt.show()
    fig.savefig(savename, bbox_inches='tight')
    return save_x, save_y
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

def plot_corrf_3d(corr, corr_pred, savename):
    """
    plot the 3D surface plot of correlation function,
    
    """
    
    time_len, space_len = corr.shape

    
    X = np.arange(space_len)            # spatial direction
    Y = np.arange(time_len)            # temporal direction
    X, Y = np.meshgrid(X, Y)           # mesh

    Z1 = corr_pred                     # predicted data
    Z2 = corr                          # experimental data

    
    fig = plt.figure(figsize=(10, 5), dpi=300)

    # Forecast
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.Blues, edgecolor='k', linewidth=0.3, antialiased=True)
    ax1.set_title('Forecast', fontsize=12)
    ax1.set_xlabel('Space index', fontsize=10)
    ax1.set_ylabel(r'Time interval $(*\Delta t)$', fontsize=10)
    ax1.set_zlabel('R', fontsize=10)
    ax1.set_zlim([0.8, 1.05])
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # Experimental
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap=cm.Oranges, edgecolor='k', linewidth=0.3, antialiased=True)
    ax2.set_title('Experimental', fontsize=12)
    ax2.set_xlabel('Space index', fontsize=10)
    ax2.set_ylabel(r'Time interval $(*\Delta t)$', fontsize=10)
    ax2.set_zlabel('R', fontsize=10)
    ax2.set_zlim([0.8, 1.05])
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.savefig(savename, bbox_inches='tight')
    plt.show()

def plot_corrf_compare(corr, corr_pred, savename):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # meshgrid for spatial and temporal axes
    X = np.arange(corr.shape[1])  # spatial index
    Y = np.arange(corr.shape[0])  # temporal index
    X, Y = np.meshgrid(X, Y)
    save_x = []
    save_y = []

    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharex=True, dpi=600)
    x_slices = np.arange(corr.shape[1])
    label_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    
    for i, ax in enumerate(axes.flat):
        x_slice = x_slices[i]
        z_slice = corr[:, x_slice]   
        y_slice = Y[:, x_slice]      

        ax.plot(y_slice, corr_pred[:, x_slice], color='blue', linewidth=1.5,
                marker='o', markersize=3, linestyle='-', label='Forecast')
        ax.plot(y_slice, z_slice, color='orange', linewidth=1,
                marker='s', markersize=3, linestyle='-.', label='Experimental data')

        
        ax.text(0.02, 0.05, label_list[i], transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left')

        
        ax.set_ylabel('R', fontsize=14)
        ax.set_xlim(0, 25)
        ax.set_ylim([0.8, 1.05])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(prop={'size': 10}, loc='upper left')
        ax.tick_params(direction='in')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        
        x_index = np.argmax(z_slice[0:7])
        max_value = np.max(z_slice[0:7])
        y_value = y_slice[x_index]
        save_x.append(max_value)
        save_y.append(y_value)

        
        ax.set_xlabel(r'Temporal interval $(*\Delta t)$', fontsize=12)

    
    plt.tight_layout()
    fig.savefig(savename, bbox_inches='tight')
    return save_x, save_y


def plot_cl(pred, target):
    """
    plot the Cl values for each pressure tap,
    comparing the forecast and experimental data.
    

    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    Cl_pred = np.sum((pred[:, 0:8] - pred[:, 13:21]), axis=1)/8
    Cl_target = np.sum((target[:, 0:8] - target[:, 13:21]), axis=1)/8
    time = np.arange(Cl_pred.shape[0])  # 时间步数
    ax.plot(time, Cl_pred, linestyle='-', label='Forecast', color='blue')
    ax.plot(time, Cl_target, linestyle='-.', label='Experimental data', color='orange')

    ax.set_xlabel(r'$\Delta T / \delta t$',  fontsize=16)
    ax.set_ylabel(r'$C_l$',  fontsize=16)
    ax.set_xlim([0, 1000])

    
    ax.legend(prop={'size': 14})
    ax.set_ylim([-1.25, 1])

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    Cl_pred_std = np.std(Cl_pred)
    Cl_target_std = np.std(Cl_target)
    Cl_max_pred =np.max(Cl_pred)
    Cl_min_pred = np.min(Cl_pred)
    Cl_max_target = np.max(Cl_target)
    Cl_min_target = np.min(Cl_target)
    print(f'Cl_pred_std: {Cl_pred_std}, Cl_target_std: {Cl_target_std}')
    print(f'Cl_pred features, min:{Cl_min_pred}, max:{Cl_max_pred}')
    print(f'Cl_target features, min:{Cl_min_target}, max:{Cl_max_target}')
    fig.savefig('Cl_plot.pdf', format='pdf', bbox_inches='tight')

def plot_cd(pred, target):
    """
    plot the Cd values for each pressure tap,
    comparing the forecast and experimental data.
    

    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    Cd_pred = np.abs(np.sum((pred[:, 8:13] - pred[:, 21:26]), axis=1)/5)
    Cd_target = np.abs(np.sum((target[:, 8:13] - target[:, 21:26]), axis=1)/5)
    time = np.arange(Cd_pred.shape[0])  # 时间步数
    ax.plot(time, Cd_pred, linestyle='-', label='Forecast', color='blue')
    ax.plot(time, Cd_target, linestyle='-.', label='Experimental data', color='orange')

    ax.set_xlabel(r'$\Delta T / \delta t$',  fontsize=16)
    ax.set_ylabel(r'$C_d$', fontsize=16)
    ax.set_xlim([0, 1000])
    ax.set_ylim([1.3, 1.9])

    
    ax.legend(prop={ 'size': 14})

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        #label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    Cd_pred_mean = np.mean(Cd_pred)
    Cd_target_mean = np.mean(Cd_target)
    Cd_min_pred = np.min(Cd_pred)
    Cd_max_pred = np.max(Cd_pred)
    Cd_min_target = np.min(Cd_target)
    Cd_max_target = np.max(Cd_target)
    print(f'Cd_pred_mean: {Cd_pred_mean}, Cd_target_mean: {Cd_target_mean}')
    print(f'Cd_pred features, min:{Cd_min_pred}, max:{Cd_max_pred}')
    print(f'Cd_target features, min:{Cd_min_target}, max:{Cd_max_target}')
    fig.savefig('Cd_plot.pdf', format='pdf', bbox_inches='tight')
    
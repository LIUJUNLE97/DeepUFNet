import numpy as np 
pred = np.load('pred_output_50.npy')
target = np.load('target_output_50.npy') 
pred_low = np.load('/scratch/junle/Unet_1_5_case/results/Unet_1/pred_output_50.npy')
# Calculate the mean squared error
from plot_results_funcs import plot_mean_std_cp, plot_fluctuation_power_spectrum, plot_cp, calculate_corrf, plot_corrf, plot_cd, plot_cl

plot_mean_std_cp(pred, target, pred_low)
plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=4, fs=400, nperseg=256)
plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=8, fs=400, nperseg=256)
#plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=10, fs=400, nperseg=256)
plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=17, fs=400, nperseg=256)
plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=21, fs=400, nperseg=256)
#plot_fluctuation_power_spectrum(pred, target, pred_low, point_No=23, fs=400, nperseg=256)
#plot_cp(pred, target, point_No=4)
#plot_cp(pred, target, point_No=8)
#plot_cp(pred, target, point_No=10)
#plot_cp(pred, target, point_No=17)
#plot_cp(pred, target, point_No=21)
#plot_cp(pred, target, point_No=23)
#plot_cd(pred, target)
#plot_cl(pred, target)


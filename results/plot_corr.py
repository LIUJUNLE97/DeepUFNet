import numpy as np 
pred = np.load('pred_output_50.npy')
target = np.load('target_output_50.npy')
# Calculate the mean squared error
from plot_results_funcs import plot_mean_std_cp, plot_fluctuation_power_spectrum, plot_cp, calculate_corrf, plot_corrf, plot_cd, plot_cl, plot_corrf_compare, plot_corrf_3d

corr_pred = calculate_corrf(nx=7, nt=700, Cp=pred[:, 13:21])
corr_target = calculate_corrf(nx=7, nt=700, Cp=target[:, 13:21])
# plot_corrf(corr=corr_pred, savename='corrf_pred.pdf')
# plot_corrf(corr=corr_target, savename='corrf_target.pdf')
plot_corrf_compare(corr=corr_target, corr_pred=corr_pred, savename='corrf_compare.pdf')
# plot_corrf_3d(corr_target, corr_pred, 'corrf_compare_3d.pdf')       
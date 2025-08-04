def plot_cp_scatter(pred, target, point_No):
    """
    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        point_No (int)
    """

    import matplotlib.pyplot as plt
    y_pred = pred[:, point_No]
    x_target = target[:, point_No]


    fig, ax = plt.subplots(figsize=(4, 4), dpi=600)


    ax.scatter(x_target, y_pred, facecolors='none', edgecolors='red')


    min_val = min(np.min(x_target), np.min(y_pred))
    max_val = max(np.max(x_target), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label=r'$y = x$')


    ax.set_xlabel('Experimental data', fontsize=14)
    ax.set_ylabel('Forecast', fontsize=14)


    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


    ax.legend(fontsize=12)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()


    save_path = f'cp_scatter_point_{point_No}.pdf'
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cp_distribution_separate(pred, target, point_No):
    """
    Input:
        pred (ndarray): shape (T, N)
        target (ndarray): shape (T, N)
        point_No (int)
    """
    
    y_pred = pred[:, point_No]
    x_target = target[:, point_No]

    
    xmin = min(np.min(x_target), np.min(y_pred))
    xmax = max(np.max(x_target), np.max(y_pred))

    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi=600, sharex=True)
    fig, ax2 = plt.subplots(figsize=(6, 4), dpi=600)

    '''
    #  =======
    bins_hist = np.arange(xmin, xmax + 0.05, 0.05)
    ax1.hist(x_target, bins=bins_hist, alpha=0.5, label='Experimental data', color='orange',
             edgecolor='black', density=True)
    ax1.hist(y_pred, bins=bins_hist, alpha=0.5, label='Forecast', color='blue',
             edgecolor='black', density=True)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.tick_params(direction='in')
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    '''
    #  =======
    sns.kdeplot(y_pred, bw_adjust=1, color='blue', linestyle='-', linewidth=1.5,
                label='Forecast', ax=ax2)
    sns.kdeplot(x_target, bw_adjust=1, color='orange', linestyle='-.', linewidth=1,
                label='Experimental data', ax=ax2)
    
    ax2.set_xlabel(r'$C_p$', fontsize=14)
    ax2.set_ylabel('PDF', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.tick_params(direction='in')
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(12)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)

    # save the figure
    plt.tight_layout()
    fig.savefig(f'cp_distribution_separate_point_{point_No}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

import numpy as np 
pred = np.load('pred_output_50.npy')
target = np.load('target_output_50.npy') 
#plot_cp_scatter(pred, target, point_No=4)
#plot_cp_scatter(pred, target, point_No=8)
#plot_cp_scatter(pred, target, point_No=17)
#plot_cp_scatter(pred, target, point_No=21)
plot_cp_distribution_separate(pred, target, point_No=4)
plot_cp_distribution_separate(pred, target, point_No=8)
plot_cp_distribution_separate(pred, target, point_No=17)
plot_cp_distribution_separate(pred, target, point_No=21)
import numpy as np 
pred = np.load('pred_output_50.npy')
target = np.load('target_output_50.npy') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def plot_cl_clip_x(pred, target):
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    Cl_pred = np.sum((pred[:, 0:8] - pred[:, 13:21]), axis=1)/8
    Cl_target = np.sum((target[:, 0:8] - target[:, 13:21]), axis=1)/8
    time = np.arange(Cl_pred.shape[0])  
    ax.plot(time, Cl_pred, linestyle='-', marker='o', label='Forecast', color='blue')
    ax.plot(time, Cl_target, linestyle='-.', marker='s', label='Experimental data', color='orange')

    ax.set_xlabel(r'$\Delta T / \delta t$', fontname='Times New Roman', fontsize=14)
    ax.set_ylabel(r'$C_l$', fontname='Times New Roman', fontsize=14)
    ax.set_xlim([0, 1000])

   
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})
    ax.set_ylim([-1.25, 1])

    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim([350, 370])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    Cl_pred_std = np.std(Cl_pred)
    Cl_target_std = np.std(Cl_target)
    print(f'Cl_pred_std: {Cl_pred_std}, Cl_target_std: {Cl_target_std}')
    fig.savefig('Cl_plot_cliped_x_2.pdf', format='pdf', bbox_inches='tight')

def plot_ins_snap_Cp(pred, target, snapid):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    Cp_pred = pred[snapid, :]
    Cp_target = target[snapid, :]
    x = np.arange(1, 27, 1)  
    ax.plot(x, Cp_pred, linestyle='-', marker= 'o', label='Forecast', color='blue')
    ax.plot(x, Cp_target, linestyle='-.', marker = 's', label='Experimental data', color='orange')
    ax.set_xlabel('Pressure tap No.', fontsize=14)
    ax.set_ylabel(r'$C_p$',  fontsize=14)
    ax.set_xlim([0.5, 26.5])
    
    ax.legend(prop={'size': 14}, loc='upper left')
    
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontname('Times New Roman')
        label.set_fontsize(14)
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    fig.savefig(f'Cp_plot_snapid_{snapid}.pdf', format='pdf', bbox_inches='tight')



plot_ins_snap_Cp(pred, target, snapid=355)
plot_ins_snap_Cp(pred, target, snapid=358)
plot_ins_snap_Cp(pred, target, snapid=367)
plot_ins_snap_Cp(pred, target, snapid=983)
plot_ins_snap_Cp(pred, target, snapid=989)
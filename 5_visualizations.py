import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
#
# This series of visualzations is for within K experiments
#
# --------------------------------------------------------------------------
def within_k_dist_matrix_procrustes(results_dir):
    """
    # handy post on not using pandas with seaborn: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
    # 
    """
    data_dict = {}
    k_list = [5, 15, 30, 50, 100, 200, 500, 750, 1000]

    x = []
    y = []
    hue = []
    for k in k_list:
        # read in data from procrustes_distances_k-<k>.txt
        k_proc_dists = np.loadtxt(os.path.join(results_dir,
                                               'procrustes_distances_k-' + str(k) + '.txt'))

        #print('for k=' + str(k) +', min=' + str(min(k_proc_dists)) + ' , mean=' + str(k_proc_dists.mean()) + ' , max=' + str(max(k_proc_dists)))

        mean = np.mean(k_proc_dists)
        std = np.std(k_proc_dists)

        print('k = ' + str(k))
        print('  min = ' + str(round(min(k_proc_dists), 4)))
        print('  mean = ' + str(round(mean, 4)))
        print('  std = ' + str(round(std, 4)))
        print('  max = ' + str(round(max(k_proc_dists), 4)))

        for dist in k_proc_dists:
            y.append(str(k))
            x.append(dist)
            hue.append('LDA')

        
        data_dict[str(k)] = k_proc_dists

    #df = pd.DataFrame(data=data_dict)


        
    ax = sns.boxplot(x=x, y=y, showfliers=False, fill=False, palette='dark')
    ax = sns.stripplot(x=x, y=y, hue=y, alpha=0.5, jitter=0.2, color="0.3")
    ax.set(xlabel='Procrustes error', ylabel='Dimensionality, $k$ ')
    
    plt.show()
    plt.close()

def across_k_30_comparison(results_dir):
    k_list = [5, 15, 30, 31, 50, 100, 500, 1000]
    x = []
    y = []

    for k in k_list:
        if k == 30:
            k_proc_dists = np.loadtxt(os.path.join(results_dir,
                                                   'procrustes_distances_k-30.txt'))
        else:
            k_proc_dists = np.loadtxt(os.path.join(results_dir,
                                                   'procrustes_distances_k-' + str(min(k, 30)) + '-' + str(max(k, 30)) + '.txt'))

        for dist in k_proc_dists:
            x.append(dist)
            y.append('k=30 vs k=' + str(k))

    f, ax = plt.subplots()

    sns.boxplot(x=x, y=y, showfliers=False, fill=False, palette='dark')
    sns.stripplot(x=x, y=y, alpha=0.5, jitter=0.2, color='0.3')
    

    plt.show()
            
def within_k_edit_distances(results_dir):
    k_list = [5, 15, 30, 50, 100, 200, 500, 750, 1000]
    x = []
    y = []

    for k in k_list:
        k_edit_dists = np.loadtxt(os.path.join(results_dir,
                                               'edit_distances_k-' + str(k) + '.txt'))

        for dist in k_edit_dists:
            y.append(str(k))
            x.append(dist)

    f, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(x=x, y=y, showfliers=False, fill=False, palette='dark')
    sns.stripplot(x=x, y=y, hue=y, alpha=0.5, jitter=0.4, color='0.3')
    #sns.swarmplot(x=x, y=y, alpha=0.5)
    plt.show()


def within_k_feat_matrix_procrustes(results_dir):
    """
    # handy post on not using pandas with seaborn: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
    # 
    """
    data_dict = {}
    k_list = [5, 15, 30, 50, 100, 200, 500, 750, 1000]

    x = []
    y = []
    hue = []
    for k in k_list:
        # read in data from procrustes_distances_k-<k>.txt
        k_proc_dists = np.loadtxt(os.path.join(results_dir,
                                               'feat_procrustes_distances_k-' + str(k) + '.txt'))

        #print('for k=' + str(k) +', min=' + str(min(k_proc_dists)) + ' , mean=' + str(k_proc_dists.mean()) + ' , max=' + str(max(k_proc_dists)))

        mean = np.mean(k_proc_dists)
        std = np.std(k_proc_dists, mean=mean)

        print('k = ' + str(k))
        print('  min = ' + str(round(min(k_proc_dists), 4)))
        print('  mean = ' + str(round(mean, 4)))
        print('  std = ' + str(round(std, 4)))
        print('  max = ' + str(round(max(k_proc_dists), 4)))

        for dist in k_proc_dists:
            y.append(str(k))
            x.append(dist)
            hue.append('LDA')

        
        data_dict[str(k)] = k_proc_dists

        
    ax = sns.boxplot(x=x, y=y, showfliers=False, fill=False, palette='dark')
    ax = sns.stripplot(x=x, y=y, hue=y, alpha=0.5, jitter=0.2, color="0.3")
    ax.set(xlabel='Procrustes disparity', ylabel='Dimensionality, k')
    ax.set_title('Topological variance from feature matrices')
    plt.show()
    plt.close()


def dist_procrustes_vs_edit_distance_scatter(results_dir):
    edit_dist_means = []
    proc_disp_means = []

    k_list = [5, 15, 30, 50, 100, 200, 500, 750, 1000]

    for k in k_list:
        
        # read in data from procrustes_distances_k-<k>.txt
        k_proc_dists = np.loadtxt(os.path.join(results_dir,
                                               'feat_procrustes_distances_k-' + str(k) + '.txt'))

        k_edit_dists = np.loadtxt(os.path.join(results_dir,
                                               'edit_distances_k-' + str(k) + '.txt'))

        proc_disp_means.append(k_proc_dists.mean())
        edit_dist_means.append(k_edit_dists.mean())

    f, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(x=edit_dist_means, y=proc_disp_means)
    
    f.show()

    
    




if __name__ == '__main__':
    
    results_dir = os.path.join(os.path.dirname(os.getcwd()), 'results_backup_20240626')
    within_k_dist_matrix_procrustes(results_dir)
   
    within_k_edit_distances(results_dir)
    across_k_30_comparison(results_dir)
    

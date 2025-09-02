"""
# zachary stine
# 2024-06-17
#
# 1. for each distance matrix that has been generated, import the 1st thousand document distances
#   a. calculate procrustes error between distance matrix and random distance matrix with same value of k
#   b. calculate procrustes error between distance matrix and each equal distance null model
"""

import os
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform


def calculate_procrustes_distances_with_null_models(sample_indices, experiment_dir, results_dir):
    """
    """

    k_list = [30, 31, 50, 100, 200, 500, 750, 1000, 15]
    dist_list = [0.5]  # [0.5, 1.0]

    for k in k_list:

        print('BEGIN k=' + str(k))
        print('-------------------------------------------------------')
        k_dir = os.path.join(experiment_dir, 'e1_k-' + str(k))

        dist_matrix_fname_list = get_dist_matrix_fnames(k_dir)

        random_matrix_fname_list = get_random_matrix_fnames(experiment_dir, k)

        random_procrustes_error_list = []
        equaldist_procrustes_error_dict = {dist: [] for dist in dist_list}

        for distmat_fname in dist_matrix_fname_list:

            distmat = squareform(np.loadtxt(distmat_fname))[sample_indices[0]:sample_indices[1], sample_indices[0]:sample_indices[1]]
            print('    file loaded: ' + str(distmat_fname))

            for random_fname in random_matrix_fname_list:
                
                rand_distmat = squareform(np.loadtxt(random_fname))
                print('        random dist matrix loaded: ' + str(random_fname))
            
                m1, m2, procrustes_error = procrustes(distmat, rand_distmat)

                random_procrustes_error_list.append(procrustes_error)

            for dist in dist_list:
                equaldistmat_fname = os.path.join(experiment_dir,
                                                  'null_models',
                                                  'equal_dist_matrix_' + str(dist).replace('.', 'd') + '.txt')

                equal_distmat = np.loadtxt(equaldistmat_fname) # This shouldn't need squareform() since it was saved directly as a matrix.
                print('        equal dist matrix loaded: ' + str(equaldistmat_fname))

                m1, m2, eq_procrustes_error = procrustes(distmat, equal_distmat)

                equaldist_procrustes_error_dict[dist].append(eq_procrustes_error)
                

        # write the k-specific procrustes errors from the random models to file
        np.savetxt(os.path.join(results_dir, 'random_procrustes_distances_k-' + str(k) + '.txt'),
                   np.array(random_procrustes_error_list))

        print('mean procrustes disparity to random models with k=' + str(k) + ' : ' + str(round(np.mean(np.array(random_procrustes_error_list)), 4)))
        print('-------------')
        print()

        # write the k-specific procrustes errors from the equal distance models to file:
        for dist in dist_list:
            np.savetxt(os.path.join(results_dir, 'equaldist_procrustes_distances_k-' + str(k) + '_' + str(dist).replace('.', 'd') + '.txt'),
                       np.array(equaldist_procrustes_error_dict[dist]))

            print('mean procrustes disparity to equal distance model with dist=' + str(dist) + ' and k=' + str(k) + ' : ' + str(round(np.mean(np.array(equaldist_procrustes_error_dict[dist])), 4)))
            print('----')
            print()
        print('END k=' + str(k))
        print('---------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------')
        print('\n')        

        
def get_random_matrix_fnames(experiment_dir, k):
    dist_matrix_fname_list = []

    null_model_dir = os.path.join(experiment_dir, 'null_models')

    for fname in os.listdir(null_model_dir):
        k_len = len(str(k))
        
        if fname[:k_len+18] == 'k-' + str(k) + '_random_distmat_':
            dist_matrix_fname_list.append(os.path.join(null_model_dir, fname))

    return dist_matrix_fname_list
        

def get_dist_matrix_fnames(k_dir):
    dist_matrix_fname_list = []
    
    for fname in os.listdir(k_dir):
        #print(fname[:8])
        
        if fname[:8] == 'distmat_':
            dist_matrix_fname_list.append(os.path.join(k_dir, fname))
    
    return dist_matrix_fname_list


if __name__ == '__main__':
    experiment_dir = os.path.join('E:')

    sample_1_indices = (0, 1000)

    results_dir = os.path.join(experiment_dir, 'exp1_results_sample-1')

    calculate_procrustes_distances_with_null_models(sample_1_indices, experiment_dir, results_dir)

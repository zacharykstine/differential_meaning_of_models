"""
# zachary stine
# 2024-06-03
#
# 1. for each value of K, calculate Procrustes error between all document jsd matrices
"""
import os
import numpy as np
import itertools
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform





    
def calculate_procrustes_distances_within_k(sample_indices, experiment_dir, results_dir):
    """
    Procrustes error is calculated between all models that have the same dimensionality, K.

    The procrustes error is calculated between the distance matrices from two different models.

    Example: There are 10 models with K=5, from which 10 distance matrices have been sampled, each
    specifying the JSD between all pairs of documents. From these 10 distance matrices, there are 45 unique pairs.
    The Procrustes error (or distance) is then calculated between each pair of distance matrices.

    How to interpret possible outcomes:
      A) If all procrustes distances are basically 0, then the randomness in a particular model does not affect meaning. In other words, the meaning
         of the documents is invariant to random initializations.

      B) If all procrustes distance are considerably large (where "large" means something only in the context of the procrustes distance to null models),
         then meaning is pretty unstable, being drastically affected by the random seed.

      C) If the mean procrustes distance for k=i is larger than the mean procrustes distance for k=j, then k=j has more semantic stability as a model choice. 
    """
    # for each value of k,
    k_list = [5, 15, 30, 31, 50, 100, 200, 500, 750, 1000]
    
    for k in k_list:
        k_dir = os.path.join(experiment_dir, 'e1_k-' + str(k))

        dist_matrix_fname_list = get_dist_matrix_fnames(k_dir)
        print(dist_matrix_fname_list)

        procrustes_error_list = []

        for fname_i, fname_j in itertools.combinations(dist_matrix_fname_list, 2):
            print(fname_i + ', ' + fname_j)
            
            distmat_i = squareform(np.loadtxt(fname_i))[sample_indices[0]:sample_indices[1], sample_indices[0]: sample_indices[1]]
            distmat_j = squareform(np.loadtxt(fname_j))[sample_indices[0]:sample_indices[1], sample_indices[0]: sample_indices[1]]

            print(distmat_i.shape)
            
            
            m1, m2, procrustes_error = procrustes(distmat_i, distmat_j)
            procrustes_error_list.append(procrustes_error)
        
        # write procrustes distances to file
        np.savetxt(os.path.join(results_dir, 'procrustes_distances_k-' + str(k) + '.txt'),
                   np.array(procrustes_error_list))

        print('mean procrustes disparity within k=' + str(k) + ' : ' + str(round(np.mean(np.array(procrustes_error_list)), 4)))
        print()

        
def calculate_procrustes_distances_across_k(experiment_dir, results_dir):
    """
    
    """
    k_list = [5, 15, 30, 31, 50, 100, 200, 500, 750, 1000]

    for k_A, k_B in itertools.combinations(k_list, 2):

        procrustes_error_list = []

        k_dir_A = os.path.join(experiment_dir, 'e1_k-' + str(k_A))
        distmat_fname_list_A = get_dist_matrix_fnames(k_dir_A)

        k_dir_B = os.path.join(experiment_dir, 'e1_k-' + str(k_B))
        distmat_fname_list_B = get_dist_matrix_fnames(k_dir_B)

        for fname_A in distmat_fname_list_A:
            distmat_A = squareform(np.loadtxt(fname_A))[sample_indices[0]:sample_indices[1], sample_indices[0]: sample_indices[1]]
            
            for fname_B in distmat_fname_list_B:
                print(fname_A + ', ' + fname_B)

                distmat_B = squareform(np.loadtxt(fname_B))[sample_indices[0]:sample_indices[1], sample_indices[0]: sample_indices[1]]

                m1, m2, procrustes_error = procrustes(distmat_A, distmat_B)
                procrustes_error_list.append(procrustes_error)
                print(procrustes_error)

        # write procrustes distances to file
        np.savetxt(os.path.join(results_dir, 'procrustes_distances_k-' + str(k_A) + '-' + str(k_B) + '.txt'),
                   np.array(procrustes_error_list))

        print('mean procrustes disparity between k=' + str(k_A)+ ' and k=' + str(k_B) + ' : ' + str(round(np.mean(np.array(procrustes_error_list)), 4)))
        print()
        

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
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    

    calculate_procrustes_distances_within_k(sample_1_indices, experiment_dir, results_dir)
    
    calculate_procrustes_distances_across_k(experiment_dir, results_dir)
    
    

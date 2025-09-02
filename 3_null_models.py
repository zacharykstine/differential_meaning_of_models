"""
# zachary stine
# 2024-06-17
#
# 1. create 10 random models by creating 1,000 random topic distributions and then calculating
#    the jsd between them to form a 1,000 by 1,000 distance matrix
#
# 2. create a null model of meaninglessness where each document is equally distant to all other
#    documents. Can do this by making a 1,000 by 1,000 distance matrix of all 0.5 (or really, any
#    number in the range [0-1]).
"""
import os
import numpy as np
from scipy.spatial.distance import pdist


def create_random_models(k_list, num_models, num_docs, output_dir):
    """
    """
    rng = np.random.default_rng()
    
    for k in k_list:
        
        for i in range(num_models):
            
            # Create 1,000 random topic distributions
            random_doc_topic_matrix = rng.integers(1, high=500, size=(num_docs, k), endpoint=True)

            '''
            print('matrix of random integers:')
            print(random_doc_topic_matrix)
            print()
            '''
            
            random_doc_topic_matrix = random_doc_topic_matrix / random_doc_topic_matrix.sum(axis=1, keepdims=True)

            '''
            print('matrix of random distributions:')
            print(random_doc_topic_matrix)
            print()
            print('shape of random distributions: ')
            print(random_doc_topic_matrix.shape)
            print()
            print('sum of a few rows: ')
            print(np.sum(random_doc_topic_matrix[0]))
            print(np.sum(random_doc_topic_matrix[5]))
            print(np.sum(random_doc_topic_matrix[42]))
            '''
            random_features_path = os.path.join(output_dir, 'k-' + str(k) + '_random_featmat_' + str(i) + '.txt')
            np.savetxt(random_features_path,
                       random_doc_topic_matrix)

            random_dist_matrix = pdist(random_doc_topic_matrix, 'jensenshannon')
            random_divg_matrix = np.square(random_dist_matrix) # converts js distance to divergence

            random_dist_path = os.path.join(output_dir, 'k-' + str(k) + '_random_distmat_' + str(i) + '.txt')
            np.savetxt(random_dist_path,
                       random_divg_matrix)
            
            
def create_equal_dist_models(dist_list, num_docs, output_dir):
    """
    """
    for dist in dist_list:
        # Create distance matrix where all values are <dist> except diagonal which should be 0s.
        dist_matrix = np.full((num_docs, num_docs), dist)
        np.fill_diagonal(dist_matrix, 0.0)

        equal_distmat_path = os.path.join(output_dir, 'equal_dist_matrix_' + str(dist).replace('.', 'd') + '.txt')
        np.savetxt(equal_distmat_path,
                   dist_matrix)
        
        
        

if __name__ == '__main__':
    experiment_dir = os.path.join('E:')

    output_dir = os.path.join(experiment_dir, 'null_models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    create_random_models([5, 15, 30, 31, 50, 100, 200, 500, 750, 1000], 10, 1000, output_dir)
    create_equal_dist_models([0.0, 0.5, 1.0], 1000, output_dir)

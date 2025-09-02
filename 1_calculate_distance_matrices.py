import os
import gensim
import numpy as np
import math
import itertools
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist


def write_and_return_topic_dists_memory_friendly(model, corpus, k, path):
    """
    :param model:
    :param corpus:
    :param path:
    :return:
    """
    doc_num = 0
    with open(path, 'w') as ofile:
        for doc in corpus:
            doc_dist = model[doc]
            doc_distribution = np.zeros(len(doc_dist), dtype='float64')
            for (topic, val) in doc_dist:
                doc_distribution[topic] = val
            np.savetxt(ofile, doc_distribution.reshape(1, k))
            #print('doc ' + str(doc_num) + ' written.')
            doc_num += 1

            
def entropy(x, checksum=True):
    if np.around(np.sum(x), decimals=6) != 1.0:
        x = x / x.sum()

    if checksum:
        # Make sure the distribution sums to something near 1.
        assert np.around(np.sum(x), decimals=6) == 1.0, 'entropy(p): p does not sum to 1.'

    # Calculate Shannon entropy, assuming 0log(0) == 0 and so can be skipped.
    h = 0.0
    for i in np.nditer(x):
        if i > 0.0:
            h += i * math.log(i, 2)
    return -h


def js_divergence(x, y, checksum=True):

    assert len(x) == len(y), 'js_divergence(): x and y do not have the same number of elements.'

    m = np.mean([x, y], axis=0, dtype=np.float64)

    h_m = entropy(m, checksum)
    h_x = entropy(x, checksum)
    h_y = entropy(y, checksum)

    return h_m - ((h_x + h_y) / 2.0)


def get_jsd_matrix(feat_matrix):
    num_docs = feat_matrix.shape[0]

    jsd_matrix = np.zeros((num_docs, num_docs))
    
    for i, j in itertools.combinations([x for x in range(num_docs)], 2):
        ij_jsd = js_divergence(feat_matrix[i], feat_matrix[j])

        jsd_matrix[i, j] = ij_jsd
        jsd_matrix[j, i] = ij_jsd

    return jsd_matrix
        

def e1(k, N, write_dir):
    '''
    # Experiment 1.1
    # - pitchfork
    # - LDA, k=30
    # - null model: each document has topic distribution of 1/k
    # - random model: randomly draw fake topic distributions
    # - Gensim implementation
    '''
    print('Experiment: K= ' + str(k) + ', N=' + str(N))
    data_name = 'pitchfork'
    data_dir = os.path.join(os.getcwd(), data_name + '_data')

    output_dir = os.path.join(write_dir, 'e1_k-' + str(k))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    passes = 15
    eval_every = None
    iterations = 500
    
    dictionary = gensim.corpora.Dictionary.load(os.path.join(data_dir, data_name + '_gensim.dict'))
    corpus = gensim.corpora.MmCorpus(os.path.join(data_dir, data_name + '_corpus.mm'))
    
    rand_state_list = np.random.choice(1000, N, replace=False)

    #
    # 1) Train N models
    #
    featmat_path_list = []
    distmat_path_list = []
    for i in range(N):
        print('  - model ' +  str(i) + ' is now training ...')
        # Train a model
        lda_model = gensim.models.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=passes, iterations=iterations,
                                           eval_every=eval_every, minimum_probability=0.0000001,
                                           alpha='auto', random_state=rand_state_list[i])

        # Save model
        lda_model.save(os.path.join(output_dir, 'lda_' + str(i)))

        # Sample theta for each document and save it to file
        docvec_path = os.path.join(output_dir, 'featmat_' + str(i))
        write_and_return_topic_dists_memory_friendly(lda_model,
                                                     corpus,
                                                     k,
                                                     docvec_path)
        featmat_path_list.append(docvec_path)

        # Calculate pairwise document distances and save distance matrix to file
        docvec_matrix = np.loadtxt(docvec_path)

        print('  - calculating distance matrix ...')

        dist_matrix = pdist(docvec_matrix, 'jensenshannon')  # Calculate JS distance, which is not quite the same as divergence
        divg_matrix = np.square(dist_matrix)  # converts JS distance to JS divergence since JS-dist = squrt(JS-divg)
        
        dist_mat_path = os.path.join(output_dir, 'distmat_' + str(i))
        np.savetxt(dist_mat_path,
                   divg_matrix)

        distmat_path_list.append(dist_mat_path)

    print('- all models in experiment are done training')
    
    #
    # 2) For each pair of models, calculate the procrustes distance for two cases:
    #   i)  Feature matrices (doc-by-topic matrices)
    #   ii) Distance matrices (doc-by-doc matrices) 
    #
    '''
    feat_pcrust_dist_list = []
    dist_pcrust_dist_list = []
    print('- pairwise distances between models is being computed...')
    
    for i, j in itertools.combinations([x for x in range(len(corpus))], 2):
        docvec_A = np.loadtxt(featmat_path_list[i])
        docvec_B = np.loadtxt(featmat_path_list[j])
        m1, m2, feat_pcrust_dist = procrustes(docvec_A, docvec_B)

        feat_pcrust_dist_list.append(feat_pcrust_dist)

        distmat_i = np.loadtxt(distmat_path_list[i])
        distmat_j = np.loadtxt(distmat_path_list[j])
        m1, m2, dist_pcrust_dist = procrustes(distmat_i, distmat_j)

        dist_pcrust_dist_list.append(dist_pcrust_dist)

    #
    # 3) Write both kinds of procrustes distances to file and compute a few stats
    #
    #
    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    np.savetxt(os.path.join(results_dir, 'e1_k-' + str(k) + '_featmat_pcrusts.txt'),
               np.array(feat_pcrust_dist_list))
    
    np.savetxt(os.path.join(results_dir, 'e1_k-' + str(k) + '_distmat_pcrusts.txt'),
               np.array(dist_pcrust_dist_list))

    
    print('K=' + str(k) + ', N=' + str(N) + '------------------')
    print()
    
    print('distmat procrustes: mean = ' + str(round(np.mean(dist_pcrust_dist_list), 5)) + ', std dev = ' + str(round(np.std(dist_pcrust_dist_list),5)))
    print(dist_pcrust_dist_list)
    print()
    
    print('featmat procrustes: mean = ' + str(round(np.mean(feat_pcrust_dist_list), 5)) + ', std dev = ' + str(round(np.std(feat_pcrust_dist_list),5)))
    print(feat_pcrust_dist_list)
    print()
    '''
    
    print('----------------------------------------\n\n')

        
        
    
if __name__ == '__main__':
    '''
    # Experiment 1.1: data=pitchfork, (model, k)=(LDA, 30)
    # - null model: each document has topic distribution of 1/k
    # - random model: randomly draw fake topic distributions
    '''
    write_dir = os.path.join('D:')
    e1(k=5, N=10, write_dir=write_dir)
    e1(k=15, N=10, write_dir=write_dir)
    e1(k=30, N=10, write_dir=write_dir)
    e1(k=50, N=10, write_dir=write_dir)
    e1(k=100, N=10, write_dir=write_dir)
    e1(k=200, N=10, write_dir=write_dir)
    e1(k=500, N=10, write_dir=write_dir)
    e1(k=750, N=10, write_dir=write_dir)
    e1(k=1000, N=10, write_dir=write_dir)
    
    
    

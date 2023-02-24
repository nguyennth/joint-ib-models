#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/12/2021
author: nhung, modified from Fenia's script https://github.com/fenchri/dsre-vae/blob/main/src/helpers/plot_tsnes.py
"""

from sklearn.manifold import TSNE #version older than scikit-learn 0.21.3 may not have this function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_mask_top10(target, top10):
    #create a mask based on top-10 labels in medmention
    mask = [True
        if x in top10
        else False
        for x in target
    ]

    return mask

alltarget = {'ncbi':['Disease'],
        'bc5cdr-disease':['Disease'],
        'bc5cdr-chemical': ['Chemical'],
        'medmention-21': ['T038', 'T103', 'T058', 'T017', 'T033', 'T082', 'T170', 'T062', 'T204','T098'],
        'shareclef': ["Disease_Disorder"],
        'genia': ["protein", "DNA", "cell_type", "cell_line", "RNA",],
        }
    
def plot_tsne_2d(data_vecs, data_labels, dataset, dim, legend, path, mode):
    """
    Plot t-SNE 2D
    data_labels should be in strings, e.g., 'Disease', not integer values
    """
    if dataset not in alltarget:
        print ("This corpus hasn't been included with target categories")
        return
    target = alltarget[dataset]
    if dataset == 'medmention-21':
        # target = ['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074','T082','T091','T092','T097','T098','T103','T168','T170','T201','T204']        
        mask = create_mask_top10(data_labels, target)
        data_labels = np.array(data_labels)[mask]
        data_vecs = np.array(data_vecs)[mask]    

    tsne_em = TSNE(n_components=2, verbose=1, random_state=42).fit_transform(np.array(data_vecs))
    
    fig = plt.figure(figsize=(8, 6))

    print('Plotting T-SNE ...')
    total_data = pd.DataFrame(list(zip(tsne_em[:, 0], tsne_em[:, 1], data_labels)),
                              columns=['dim1', 'dim2', 'NE category'])
    
    sns.set_style("whitegrid")
    sns.set_context("paper")
    
    ax = sns.scatterplot(x="dim1", y="dim2", data=total_data, hue='NE category', hue_order=target, linewidth=0,
                         palette='deep')

    ax.grid(b=True, which='major', color='lightgrey', linewidth=0.5)

    plt.xlabel("")
    plt.ylabel("")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        plt.setp(ax.get_legend().get_texts(), fontsize='8')
        plt.legend(loc='lower right')
    else:
        ax.get_legend().remove()

    # fig.savefig(f'{path}/tsne_{dataset}_{dim}d_{mode}_2D.png', bbox_inches='tight')
    # fig.savefig(f'{path}/tsne_{dataset}_{dim}d_{mode}_2D.pdf', bbox_inches='tight')
    # print(f'Figure saved at {path}/tsne_{dataset}_{dim}d_{mode}_2D.png')
    
    fig.savefig(f'{path}/{dataset}_{dim}d_{mode}_2D.png', bbox_inches='tight')    
    print(f'Figure saved at {path}/tsne_{dataset}_{dim}d_{mode}_2D.png')
    
import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

import pickle as pkl
import hashlib

import os
import logging

def expand_descriptors(descs, db=None, alpha=0, k=0):
    assert k >= 0 and alpha >= 0, 'k and alpha must be non-negative'
    if k == 0:
        return descs
    descs = tonumpy(descs)
    n = descs.shape[0]
    db_descs = tonumpy(db if db is not None else descs)

    sim = matmul(descs, db_descs)
    if db is None:
        sim[np.diag_indices(n)] = 0

    idx = np.argpartition(sim, int(-k), axis=1)[:, int(-k):]
    descs_aug = np.zeros_like(descs)
    for i in range(n):
        new_q = np.vstack([db_descs[j, :] * sim[i, j]**alpha for j in idx[i]])
        new_q = np.vstack([descs[i], new_q])
        new_q = np.mean(new_q, axis=0)
        descs_aug[i] = new_q / np.linalg.norm(new_q)
    return descs_aug


def pairs_from_dir(images_path,data_sorted,topN = 10,output_pairs):

    logging.info('Exporting image pairs from DIR ...')
    pairs = []
    listdata = os.listdir(images_path)
    counter = 0

    images_list = []
    extension = ['.png','.jpg','.bmp']
    for image_name in listdata:
        fmt = image_name[-4:].lower()
        if fmt in extension:
            images_list.append(image_name)

    for image_name in images_list:
        top_simi_id = data_sorted(counter,0:topN)
        for id in top_simi_id:
            pair = (image_name,images_list[id])
            pairs.append(pair) 
        counter += 1           
    logging.info(f'Found {len(pairs)} pairs.')
    with open(output_pairs,'w') as f:
        f.write('\n'.join(' '.join([i,j]) for i,j in pairs))
    

def eval_model(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
               aqe=None, adba=None, threads=8, batch_size=16, save_feats=None,
               load_feats=None, dbg=()):
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Evaluation...")
    query_db = db.get_query_db()

    # load DB feats
    bdescs = []
    qdescs = []

    bdescs = np.load(os.path.join(load_feats, 'feats.bdescs.npy'))
    qdescs = bdescs 

    if whiten is not None:
        bdescs = common.whiten_features(tonumpy(bdescs), net.pca, **whiten)
        qdescs = common.whiten_features(tonumpy(qdescs), net.pca, **whiten)

    if adba is not None:
        bdescs = expand_descriptors(bdescs, **args.adba)
    if aqe is not None:
        qdescs = expand_descriptors(qdescs, db=bdescs, **args.aqe)

    scores = matmul(qdescs, bdescs)
    data_sorted = np.argsort(-scores)

    del bdescs
    del qdescs

    return data_sorted


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default='Landmarks_clean', help='applies whitening')

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    # write image pairs
    output_pairs
    parser.add_argument('--images_path', type=str, default='', help='path to input images path')
    parser.add_argument('--output_pairs', type=str, default='' help='path to output pairs')
    parser.add_argument('--topN', type=int, default=20, help='top N most similar images')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

    dl.download_dataset(args.dataset)

    dataset = datasets.create(args.dataset)
    print("Test dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    data_sorted = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                  threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                  save_feats=args.save_feats, load_feats=args.load_feats)
    
    # Write image_pairs, used as SFM match-list
    pairs_from_dir( images_path=args.images_path,
                    data_sorted = data_sorted,
                    topN = args.topN,
                    output_pairs = args.output_pairs)

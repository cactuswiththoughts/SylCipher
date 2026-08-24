# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans_FC

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    fix_first_n_clus,
    verbose
):
    return MiniBatchKMeans_FC(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=verbose,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        fix_first_n_clus=fix_first_n_clus,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path,
    km_path_phase_1,
    km_path_fixed,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
    verbose
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    phase_1_km_clus = np.load(km_path_phase_1)
    fixed_km_clus = np.load(km_path_fixed)

    logger.info(f'Checking two km matrix shape: {phase_1_km_clus.shape}; {fixed_km_clus.shape}')
    pairwise_dist = np.power(np.linalg.norm(phase_1_km_clus[:, None, :] - fixed_km_clus[None, :, :], axis=-1), 2)
    mean_pairwise_dist = np.mean(pairwise_dist, axis = 1) 
    logger.info(f'Checking distance shape: {mean_pairwise_dist.shape}; {pairwise_dist.shape}')
    mean_dist_to_fixed_idxs = np.argsort(mean_pairwise_dist)[::-1][:phase_1_km_clus.shape[0] - fixed_km_clus.shape[0]]

    logger.info(f'Checking if the correct distances are selected: {len(mean_dist_to_fixed_idxs)}; {mean_pairwise_dist[mean_dist_to_fixed_idxs]}')
    non_fixed_clus_init = np.stack([phase_1_km_clus[cidx] for cidx in mean_dist_to_fixed_idxs], axis = 0)
    
    init_clus_centers = np.concatenate([fixed_km_clus, non_fixed_clus_init], axis = 0)
    fix_first_n_clus = fixed_km_clus.shape[0]

    km_model = get_km_model(
        n_clusters,
        init_clus_centers,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
        fix_first_n_clus,
        verbose
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("km_path", type=str)
    parser.add_argument("km_path_phase_1", type=str)
    parser.add_argument("km_path_fixed", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    #parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--batch_size", default=1000000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    parser.add_argument("--verbose", default=1, type=int)
    args = parser.parse_args()
    logging.info(str(args))
    
    km_dir = os.path.dirname(args.km_path)
    os.makedirs(km_dir, exist_ok=True)

    learn_kmeans(**vars(args))

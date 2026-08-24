# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os.path as op
import re
from tabulate import tabulate
from collections import Counter


def read_lab(path):
    uid2labs = {}
    idx = 0
    with open(path) as f:
        for line in f:
            labs = line.strip().split()
            uid2labs[idx] = labs
            idx += 1
    # print(path)
    # print(uid2labs[0])
    # print(idx)
    return uid2labs


def comp_purity(p_xy, axis):
    max_p = p_xy.max(axis=axis)
    marg_p = p_xy.sum(axis=axis)
    indv_pur = max_p / marg_p
    aggr_pur = max_p.sum()
    return indv_pur, aggr_pur


def comp_entropy(p):
    return (-p * np.log(p + 1e-8)).sum()


def comp_norm_mutual_info(p_xy):
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    mi = (p_xy * pmi).sum()
    h_x = comp_entropy(p_x)
    h_y = comp_entropy(p_y)
    return mi, mi / h_x, mi / h_y, h_x, h_y


def comp_joint_prob(uid2refs, uid2hyps):
    """
    Args:
        pad: padding for spliced-feature derived labels
    """
    cnts = Counter()
    skipped = []
    abs_frmdiff = 0
    for uid in uid2refs:
        if uid not in uid2hyps:
            skipped.append(uid)
            continue
        refs = uid2refs[uid]
        hyps = uid2hyps[uid]
        abs_frmdiff += abs(len(refs) - len(hyps))
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]
        cnts.update(zip(refs, hyps))
    tot = sum(cnts.values())

    ref_set = sorted({ref for ref, _ in cnts.keys()})
    hyp_set = sorted({hyp for _, hyp in cnts.keys()})
    ref2pid = dict(zip(ref_set, range(len(ref_set))))
    hyp2lid = dict(zip(hyp_set, range(len(hyp_set))))
    # print(hyp_set)
    p_xy = np.zeros((len(ref2pid), len(hyp2lid)), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        p_xy[ref2pid[ref], hyp2lid[hyp]] = cnt
    p_xy /= p_xy.sum()
    return p_xy, ref2pid, hyp2lid, tot, abs_frmdiff, skipped


def _main(uid2refs, uid2hyps, which_set, which_modality, which_km_modality, which_layer, verbose):
    (p_xy, ref2pid, hyp2lid, tot, frmdiff, skipped) = comp_joint_prob(
        uid2refs, uid2hyps
    )
    ref_pur_by_hyp, ref_pur = comp_purity(p_xy, axis=0)
    hyp_pur_by_ref, hyp_pur = comp_purity(p_xy, axis=1)
    (mi, mi_norm_by_ref, mi_norm_by_hyp, h_ref, h_hyp) = comp_norm_mutual_info(p_xy)
    outputs = {
        "which set": which_set,
        "modality": which_modality,
        "k-means modality": which_km_modality,
        "layer #": which_layer,
        "ref pur": ref_pur,
        "hyp pur": hyp_pur,
        "H(ref)": h_ref,
        "H(hyp)": h_hyp,
        "MI": mi,
        "MI/H(ref)": mi_norm_by_ref,
        "p_xy shape": p_xy.shape,
        "frm tot": tot,
        "frm diff": frmdiff,
        "utt tot": len(uid2refs),
        "utt miss": len(skipped),
    }
    print(outputs)
    # print(tabulate([outputs.values()], outputs.keys(), floatfmt=".4f"))


if __name__ == "__main__":
    """
    compute quality of labels with respect to phone or another labels if set
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("lab_dir")
    parser.add_argument("lab_name")
    parser.add_argument("layer_name")
    parser.add_argument("modality_name")
    parser.add_argument("km_modality_name")
    
    
    args = parser.parse_args()
    
    uid2refs = read_lab(op.join(args.lab_dir, f'{args.lab_name}_{args.modality_name}_layer_{args.layer_name}_0_1.labels'))
    uid2hyps = read_lab(op.join(args.lab_dir, f'labels_layer_{args.layer_name}_from_{args.km_modality_name}_km',f'{args.lab_name}_{args.modality_name}_layer_{args.layer_name}_0_1.km'))
    
    _main(uid2refs, uid2hyps, which_set = args.lab_name, which_modality = args.modality_name,which_km_modality = args.km_modality_name, which_layer = args.layer_name, verbose=True)
    
    
    

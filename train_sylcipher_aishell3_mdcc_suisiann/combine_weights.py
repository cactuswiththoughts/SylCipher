import argparse
import editdistance
import json
import logging
import numpy as np
import os
import pickle as pkl
import sys
import torch
from copy import deepcopy
from pathlib import Path
from shutil import copyfile


logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('combine_weights')


class ApplyKmeans(object):
    def __init__(self, C_np):
        self.C_np = C_np.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def combine_checkpoints(ckpt_paths, out_dir, out_ckpt_path, n_special=4):
    ckpts = [torch.load(cp) for cp in ckpt_paths]
    new_ckpt = deepcopy(ckpts[0])
    # Transfer weights for the speech prenet and postnet
    speech_keys = [
        'encoder.src_proj.weight',
        'encoder.src_proj.bias',
        'speech_encoder_prenet.encoder_prenet.0.weight',
        'speech_predictor.predictor_net.weight',
        'speech_predictor.predictor_net.bias',
        'speech_encoder_postnet.label_embs_concat',
        'speech_decoder_prenet.embed_tokens.weight',
        'speech_decoder_postnet.output_projection.weight',
    ]

    new_ckpt['model']['speech_encoder_prenet.quantizer.embedding'] = torch.from_numpy(np.load(str(out_dir / 'seg_feats' / 'km_model.npy')))
    new_ckpt['model']['speech_encoder_prenet.quantizer.embedding.data'] = torch.from_numpy(np.load(str(out_dir / 'seg_feats' / 'km_model.npy')))
    new_ckpt['cfg']['task'].data = str(out_dir)
    new_ckpt['cfg']['model'].init_clus_dir = str(out_dir / 'seg_feats' / 'km_model.npy')
    new_ckpt['cfg']['model'].init_model_dir = out_ckpt_path
    for k in speech_keys:
        p = torch.cat([ckpt['model'][k][n_special:-1] for ckpt in ckpts])
        p = torch.cat([ckpts[0]['model'][k][:n_special], p, ckpts[0]['model'][k][-1:]])
        new_ckpt['model'][k] = p
        print(k, p.shape)

    text_keys = [
        'encoder.proj.weight',
        'encoder.proj.bias',
        'text_encoder_prenet.encoder_prenet.0.weight',
        'text_predictor.predictor_net.weight',
        'text_predictor.predictor_net.bias',
        'text_encoder_postnet.label_embs_concat',
        'text_decoder_prenet.embed_tokens.weight',
        'text_decoder_postnet.output_projection.weight',
    ]

    vocab = []
    row_indices = []
    i = 0
    for src_dir in src_dirs:
        with open(src_dir / 'dict.txt', 'r') as f:
            for l in f:
                w, c = l.strip().split()
                if not w in vocab:
                    vocab.append(w)
                    row_indices.append(i)
                i += 1

    for k in text_keys:
        p = torch.cat([ckpt['model'][k][n_special:] for ckpt in ckpts])
        p = p[row_indices]
        p = torch.cat([ckpts[0]['model'][k][:n_special], p, ckpts[0]['model'][k][-2:]])
        new_ckpt['model'][k] = p
        print(k, p.shape)
    
    Path(out_ckpt_path).parent.mkdir(parents=True, exist_ok=True)   
    torch.save(new_ckpt, out_ckpt_path)


def combine_dict(src_dirs, out_dir):
    km_size = np.load(out_dir / 'seg_feats' / 'km_model.npy').shape[0]
    offset = 0
    with open(out_dir / 'dict.audio.txt', 'w') as f_d1,\
        open(out_dir / 'dict.audio0.txt', 'w') as f_d2:
        for i in range(km_size):
            print(f'{i} 1', file=f_d1)
            print(f'{i} 1', file=f_d2)
#        for src_dir in src_dirs:
#            with open(src_dir / 'dict.audio.txt', 'r') as f:
#                lines = f.read().strip().split('\n')
#                for l in lines:
#                    u = int(l.split()[0])
#                    print(f'{offset+u} 1', file=f_d1)
#                    print(f'{offset+u} 1', file=f_d2)
#                offset += len(lines)

    offset = 0
    with open(out_dir / 'dict.txt', 'w') as f_d:
        vocab = []
        for src_dir in src_dirs:
            with open(src_dir / 'dict.txt', 'r') as f:
                for l in f:
                    w, c = l.strip().split()
                    if not w in vocab:
                        print(l.strip(), file=f_d)
                        vocab.append(w)

        print(f'Combined vocab size: {len(vocab)}')
        logger.info(f'Combined vocab size: {len(vocab)}')


def reassign_cluster(src_dirs, out_dir):
    # Concat centroids
    C = np.concatenate(
        [np.load(str(src_dir / 'seg_feats' / 'km_model.npy')) for src_dir in src_dirs]
    )
    logger.info(f'centroid size: {C.shape}')
    apply_km = ApplyKmeans(C)
    (out_dir / 'seg_feats').mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'seg_feats' / 'km_model.npy'), C)
    Path(out_dir / 'discrete_speech_train_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)
    for i in range(len(src_dirs)):
        Path(out_dir / 'discrete_speech_test{i}_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)

    vocab = []
    with open(out_dir / 'discrete_speech_train_clsattnbndkmeans' / 'data.km', 'w') as f_train:
        for i, src_dir in enumerate(src_dirs):
            # Train
            seg_feats = torch.from_numpy(
                np.load(str(src_dir / 'seg_feats' / 'train_0_1.npy')).astype(np.float32)
            )
            with open(src_dir / 'seg_feats' / 'train_0_1.len', 'r') as f:
                lines = f.read().strip().split('\n')
                sizes = list(map(int, lines))

            labels = apply_km(seg_feats).tolist()
            labels = list(map(str, labels))
            print(f'# train labels: {len(labels)}, total train sizes: {sum(sizes)}')  # XXX

            offset = 0
            for size in sizes:
                print(' '.join(labels[offset:offset+size]), file=f_train)
                offset += size

            # Test
            seg_feats = torch.from_numpy(
                np.load(str(src_dir / 'seg_feats' / 'test_0_1.npy')).astype(np.float32)
            )
            with open(src_dir / 'seg_feats' / 'test_0_1.len', 'r') as f:
                lines = f.read().strip().split('\n')
                sizes = list(map(int, lines))

            labels = apply_km(seg_feats).tolist()
            labels = list(map(str, labels))
            print(f'# test labels: {len(labels)}, total test sizes: {sum(sizes)}')  # XXX
            vocab.extend(list(set(labels)))
            offset = 0
            (out_dir / f'discrete_speech_test{i}_clsattnbndkmeans').mkdir(exist_ok=True, parents=True)
            with open(out_dir / f'discrete_speech_test{i}_clsattnbndkmeans' / 'data.km', 'w') as f_test:
                for size in sizes: 
                    print(' '.join(labels[offset:offset+size]), file=f_test)
                    offset += size
    logger.info(f'vocab size: {len(set(vocab))}')


def reassign_frame_cluster(src_dirs, out_dir):
    out_train_dir = out_dir / 'discrete_speech_train_clsattnbndkmeans'
    km_offset = 0
    with open(out_train_dir / 'data_frame_clus.km', 'w') as f_tr:
        for i, src_dir in enumerate(src_dirs):
            train_dir = src_dir / 'discrete_speech_train_clsattnbndkmeans'
            feat_name = '_'.join(src_dir.parent.parent.name.split('_')[:3])
            C = np.load(str(src_dir / '../../..' / feat_name / 'km_model.npy'))
            km_size = C.shape[0]

            with open(train_dir / 'data_frame_clus.km', 'r') as f_tr_in:
                lines = f_tr_in.read().strip().split('\n')
                for l in lines:
                    labels = ' '.join([str(int(y)+km_offset) for y in l.split()])
                    print(labels, file=f_tr)

            test_dir = src_dir / 'discrete_speech_test_clsattnbndkmeans'
            out_test_dir = out_dir / f'discrete_speech_test{i}_clsattnbndkmeans'
            with open(test_dir / 'data_frame_clus.km', 'r') as f_te_in,\
                open(out_test_dir / 'data_frame_clus.km', 'w') as f_te:
                lines = f_te_in.read().strip().split('\n')
                for l in lines:
                    labels = ' '.join([str(int(y)+km_offset) for y in l.split()])
                    print(labels, file=f_te)
            km_offset += km_size
    print(f'Total number of frame clusters: {km_offset}')


def combine_metadata(src_dirs, out_dir):
    bnd_errors = []
    bound_dict = {}
    meta_dict = {}
    split = 'discrete_speech_train_clsattnbndkmeans'
    with open(out_dir / split / 'file_list.txt', 'w') as f_list,\
        open(out_dir / split / 'bnd_errors.txt', 'w') as f_bnd_err,\
        open(out_dir / split / 'data_dict_bnd.pkl', 'wb') as f_bnd,\
        open(out_dir / split / 'meta_data.json', 'w') as f_meta,\
        open(out_dir / split / 'data.lengths', 'w') as f_len:
        for src_dir in src_dirs:
            with open(src_dir / split / 'file_list.txt', 'r') as f:
                print(f.read().strip(), file=f_list)
            
            with open(src_dir / split / 'bnd_errors.txt', 'r') as f:
                if len(f.read.strip()):
                    print(f.read().strip(), file=f_bnd_err)

            with open(src_dir / split / 'data_dict_bnd.pkl', 'rb') as f:
                cur_bound_dict = pkl.load(f)
                bound_dict.update(cur_bound_dict)

            with open(src_dir / split / 'meta_data.json', 'r') as f:
                cur_meta_dict = json.load(f)
                meta_dict.update(cur_meta_dict)

            with open(src_dir / split / 'data.lengths', 'r') as f:
                lines = f.read().strip()
                print(lines, file=f_len)

        pkl.dump(bound_dict, f_bnd)
        json.dump(meta_dict, f_meta, indent=2)
    
    for i, src_dir in enumerate(src_dirs):
        split = 'discrete_speech_test_clsattnbndkmeans'
        out_split = f'discrete_speech_test{i}_clsattnbndkmeans'
        copyfile(str(src_dir / split / 'file_list.txt'), str(out_dir / out_split / 'file_list.txt'))
        copyfile(str(src_dir / split / 'bnd_errors.txt'), str(out_dir / out_split / 'bnd_errors.txt'))
        copyfile(str(src_dir / split / 'data_dict_bnd.pkl'), str(out_dir / out_split / 'data_dict_bnd.pkl'))
        copyfile(str(src_dir / split / 'meta_data.json'), str(out_dir / out_split / 'meta_data.json'))
        copyfile(str(src_dir / split / 'data.lengths'), str(out_dir / out_split / 'data.lengths'))


def combine_text_data(src_dirs, out_dir):
    split = 'train'
    with open(out_dir / f'../{split}_words.txt', 'w') as f_tr:
        for src_dir in src_dirs:
            print(src_dir / f'../{split}_words.txt')
            with open(src_dir / f'../{split}_words.txt', 'r') as f:
                print(f.read().strip(), file=f_tr)

    split = 'test'
    for i, src_dir in enumerate(src_dirs):
        with open(src_dir / f'../{split}_words.txt', 'r') as f,\
            open(out_dir / f'../{split}{i}_words.txt', 'w') as f_te:
            print(f.read().strip(), file=f_te)


def _manifest_root():
    env = os.environ.get('MANIFEST_ROOT')
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / 'manifest'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge AISHELL-3 / MDCC / SuiSiann SylCipher checkpoints into a shared codebook.'
    )
    mroot = _manifest_root()
    parser.add_argument(
        '--src-dirs', nargs=3, default=[
            str(mroot / 'aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048'),
            str(mroot / 'mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608'),
            str(mroot / 'suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024'),
        ],
        help='km_dir for AISHELL-3, MDCC, SuiSiann (in that order)',
    )
    parser.add_argument(
        '--ckpt-paths', nargs=3, default=[
            str(mroot / 'aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/checkpoint_best.pt'),
            str(mroot / 'mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608/sylcipher_transfer_aishell_3/checkpoint_best.pt'),
            str(mroot / 'suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024/sylcipher_transfer_aishell_3/checkpoint_best.pt'),
        ],
        help='checkpoint_best.pt for AISHELL-3, MDCC, SuiSiann (in that order)',
    )
    parser.add_argument(
        '--out-dir',
        default=str(mroot / 'aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048'),
    )
    parser.add_argument(
        '--out-ckpt-path',
        default=str(mroot / 'aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048/sylcipher_transfer_aishell3_mdcc_suisiann/checkpoint_best.pt'),
    )
    args = parser.parse_args()

    src_dirs = [Path(p) for p in args.src_dirs]
    ckpt_paths = args.ckpt_paths
    out_dir = Path(args.out_dir)
    out_ckpt_path = args.out_ckpt_path

    logger.info("Combine checkpoints")
    combine_checkpoints(ckpt_paths, out_dir, out_ckpt_path)

    logger.info("Re-assign cluster units")
    # reassign_cluster(src_dirs, out_dir)
    # reassign_frame_cluster(src_dirs, out_dir)

    logger.info("Create new dictionaries")
    # combine_dict(src_dirs, out_dir)

    logger.info("Concat filelist, bnd_errors, boundaries, etc.")
    # combine_metadata(src_dirs, out_dir)

    logger.info("Combine text data")
    # combine_text_data(src_dirs, out_dir)

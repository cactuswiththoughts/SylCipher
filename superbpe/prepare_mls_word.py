import argparse
import json
from tqdm import tqdm
import logging
import os
import re
import sys
import random
from pathlib import Path


VOWELS = "aeiouyäöüáàâãéêíóôõúáéíóúüàâæéèêëîïôœùûüàèéìòóùąęó"
logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)
logger = logging.getLogger('prepare_mls_word')


def read_transcripts(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            utt, *txt = line.strip().split()
            d[utt] = txt
    return d


def read_segments(path):
    segs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            utt, spk, start, end = line.strip().split()
            dur = float(end) - float(start)
            segs.append((utt, dur))
    return segs


def process_language(lang_dir, filelist, out_root):
    lang = lang_dir.name
    logger.info(f"\n========= {lang} ==========")

    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = open(out_dir / f'lm.upper.lid.txt', 'w', encoding='utf-8')
    vocab = []
    total = 0
    for file in filelist:
        meta_dict = {}
        split = Path(file).stem.split('_')[0]
        with open(file, 'r') as f:
            file_name_list = [line.strip() for line in f.readlines()]

        transcripts = read_transcripts(lang_dir / split / "transcripts.txt")
        segments = read_segments(lang_dir / split / "segments.txt")

        fmeta = open(out_dir / f'{split}_gradseg_style.json', 'w', encoding='utf-8')
        for utt, dur in tqdm(segments):
            if utt not in transcripts:
                continue

            part1, part2 = utt.split('_')[:2]
            fn = f'{split}/audio/{part1}/{part2}/{utt}.flac'
            if fn not in file_name_list:
                continue
            text = transcripts[utt]
            print(' '.join(text), file=font)
            words = []
            nframes = int(dur * 16000)
            n_words = len(text)
            seg_nframes = nframes // n_words
            for i, w in enumerate(text):
                words.append([i*seg_nframes, (i+1)*seg_nframes, w])
                if not w in vocab:
                    vocab.append(w) 
            meta_dict[fn] = words

            total += dur
        
        json.dump(meta_dict, fmeta, indent=2, ensure_ascii=False)
        fmeta.close()
    
    font.close()

    logger.info(f'Vocab size: {len(vocab)}')
    logger.info(f'Collected {total/3600:.2f} hours')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mls_root", default='/data/sls/scratch/limingw/data/MLS/mls_german',
                        help="root folder containing mls_english, mls_german, ...")
    parser.add_argument("--filelist", nargs='+', default=['/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_german_vad/train_file_list.txt','/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_german_vad/test_file_list.txt'])
    parser.add_argument("--out_dir", default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls_word/mls_german',
                        help="output folder")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    root = Path(args.mls_root)
 
    process_language(root, args.filelist, args.out_dir)
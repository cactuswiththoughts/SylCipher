import argparse
import json
import logging
import os
import re
import sys
import random
from pathlib import Path
import pyphen
from tqdm import tqdm

VOWELS = "aeiouyäöüáàâãéêíóôõúáéíóúüàâæéèêëîïôœùûüàèéìòóùąęó"
logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)
logger = logging.getLogger('prepare_mls')


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


def has_vowel(seg):
    for v in VOWELS:
        if v in seg:
            return True
    return False


def merge_novowel_segments(segs):
    new_segs = []
    n_seg = len(segs)
    i = 0
    while i < n_seg:
        seg = segs[i]
        if not has_vowel(seg):
            # If seg does not contain vowels and is at the end, add it to the previous seg
            if i == n_seg - 1:
                if len(new_segs):
                    new_segs[-1] += seg
                else:
                    new_segs = [seg]
            # Otherwise, add it to the next seg
            else:
                new_segs.append(seg+segs[i+1])
                i += 1
        i += 1
    return new_segs


def naive_syllabify(w):
    w0 = w
    # Drop a truly silent 'e' (unless -le ending)    
    if w.endswith('e') and not re.search(rf'[^ {VOWELS}]le$', w):
        w = w[:-1]    

    # For 'ed' endings pronounced as /t/ or /d/, strip the 'e'
    #     (i.e., drop the e in -ed unless the root ends in 't' or 'd')
    if w.endswith("ed") and len(w) > 2 and w[-3] not in ("t","d"):
        # e.g. 'jumped' → 'jumpd', 'baked' → 'bakd'
        w = w[:-2] + w[-1]

    # For 3+ consonants between vowels, split V C-CC… V → VC-CC…V
    w = re.sub(
        rf"(?<=[{VOWELS}][^{VOWELS}])"      # look-behind: V + C
        rf"(?=[^ {VOWELS}]"                # look-ahead start: C
        rf"{{2,}}[{VOWELS}])",             # at least 2 more C’s, then V
        "-",
        w
    )

    # VCCV -> VC | CV
    w1 = re.sub(rf'(?<=[{VOWELS}][^{VOWELS}])(?=[^{VOWELS}][{VOWELS}])', '-', w.lower())
    # VCV -> V | CV   
    w2 = re.sub(rf'(?<=[{VOWELS}])(?=[^{VOWELS}][{VOWELS}])', '-', w1)

    if w0 != w:
        if w0.endswith('e'):
            w2 += 'e'
        elif w0.endswith('ed'):
            w2 = w2[:-1] + 'ed'
    return w2.split('-')


def is_non_word(token):
    token = token.lower()

    if any(c.isdigit() for c in token):
        return True

    if token.isupper() and len(token) <= 5:
        return True

    if sum(c in VOWELS for c in token) == 0:
        return True

    #if len(token) <= 2:
    #    return True   # uh, mm, etc.

    return False


def syllabify(text, dur, dic):
    nframes = int(dur * 16000)
    syls = []
    wrd2syl = {}
    for word in text:
        word = word.lower()
        if is_non_word(word):
            logger.info(f'Found non-word: {word}')
            continue

        if word not in wrd2syl:
            seg = word
            syls_ = dic.inserted(seg).split('-')
            if len(syls_) == 1:
                syls_ = naive_syllabify(seg)
                logger.info(f'Naive: {word} -> {syls_}')
            else:
                new_syls_ = []
                for syl in syls_:
                    new_syls_.extend(naive_syllabify(syl))
                syls_ = new_syls_
                logger.info(f'{word} -> {syls_}')
            syls.extend(syls_)
            wrd2syl[word] = syls_
        else:
            syls.extend(wrd2syl[word])
    
    out = []
    seg_nframes = nframes // max(len(syls), 1)
    for i, syl in enumerate(syls):
        out.append([i*seg_nframes, (i+1)*seg_nframes, syl])

    return out


# -------------------------------------------------
# main logic
# -------------------------------------------------

def process_language(lang_dir, split, out_root, hours, seed):
    lang = lang_dir.name
    logger.info(f"\n========= {lang} ==========")

    transcripts = read_transcripts(lang_dir / split / "transcripts.txt")
    segments = read_segments(lang_dir / split / "segments.txt")

    random.seed(seed)
    random.shuffle(segments)

    lang_map = {
        "mls_english": "en_US",
        "mls_german": "de_DE",
        "mls_french": "fr_FR",
        "mls_spanish": "es_ES",
        "mls_portuguese": "pt_PT",
        "mls_polish": "pl_PL",
        "mls_italian": "it_IT",
        "mls_dutch": "nl_NL",
    }

    dic = pyphen.Pyphen(lang=lang_map.get(lang, "en_US"))

    out_dir = Path(out_root) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    font = open(out_dir / f'{split}.tsv', 'w', encoding='utf-8')
    font.write(f'{lang_dir}\n')

    meta_dict = {}
    fmeta = open(out_dir / f'{split}_gradseg_style.json', 'w', encoding='utf-8')

    target = hours * 3600
    total = 0

    vocab = []
    for utt, dur in tqdm(segments):
        if total >= target:
            break

        if utt not in transcripts:
            continue

        text = transcripts[utt]
        syl = syllabify(text, dur, dic)
        for _, _, syl_ in syl:
            if not syl_ in vocab:
                vocab.append(syl_) 

        part1, part2 = utt.split('_')[:2]
        fn = f'{split}/audio/{part1}/{part2}/{utt}.flac'
        meta_dict[fn] = syl
        font.write(f'{fn}\t{int(dur*16000)}\n')

        total += dur

    json.dump(meta_dict, fmeta, indent=2, ensure_ascii=False)
    fmeta.close()
    font.close()

    logger.info(f'Vocab size: {len(vocab)}')
    logger.info(f'Collected {total/3600:.2f} hours')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mls_root", default='/data/sls/scratch/limingw/data/MLS',
                        help="root folder containing mls_english, mls_german, ...")
    parser.add_argument("--hours", type=float, default=100)
    parser.add_argument("--out_dir", default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls',
                        help="output folder")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    root = Path(args.mls_root)

    for lang_dir in root.glob("mls_*"):
        for split in ['train', 'test']:
            process_language(lang_dir, split, args.out_dir, args.hours, args.seed)

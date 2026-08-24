import ast
import argparse
from copy import deepcopy
import pandas as pd
import json
import re
from pathlib import Path
import torchaudio
from tqdm import tqdm
VOWELS = "aeiou"
NASALS = ["m", "n"]

import re
import unicodedata

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def is_vowel(ch, vowel_set):
    """Check if ch is in language vowel set or is a Unicode vowel-like char."""
    if ch in vowel_set:
        return True
    # fallback: many languages use common vowels even if not listed
    return ch in "aeiouáéíóúàèìòùâêîôûäëïöüãõɑɛɪɔʊœøå"


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[^a-zA-Záéíóúàèìòùâêîôûäëïöüãõɑɛɪɔʊœøåçñɲŋʼʔ\s]", " ", text)


# -----------------------------------------------------------
# Generic maximal-onset universal syllabifier
# -----------------------------------------------------------

def universal_syllabify(word, vowels="aeiou"):
    """
    A fallback syllabifier using maximal-onset principle.
    Works for 102 languages but is less accurate than language-specific rules.
    """
    syllables = []
    i = 0
    w = word.lower()

    while i < len(w):
        # Find nucleus
        onset_start = i 
        while i < len(w) and not is_vowel(w[i], vowels):
            i += 1

        # Gather following consonants until next vowel, but stop before last consonant group
        coda_end = i
        while coda_end < len(w) and is_vowel(w[coda_end], vowels):
            coda_end += 1
        i = coda_end

        # Optional consonant coda
        if i + 1 == len(w):
            if w[i] not in vowels:
                coda_end += 1
        i = coda_end

        # Decide syllable boundary using maximal onset:
        # we take consonants before the NEXT vowel as onset of next syllable
        syllables.append(w[onset_start:coda_end])

    return syllables


# -----------------------------------------------------------
# Language-specific syllabifiers dictionary
# -----------------------------------------------------------
def normalize_shona_prefixes(word):
    w = word.lower()
    if w.startswith('zv'):
        # zvi- / zve- / zva- → zv-
        w = "zv" + w[3:]
    return w

def syllabify_bantu(word, vowels="aeiou", prenasal_clusters=("mb", "nd", "ng", "nj")):
    """Good for Luganda, Shona, Nyanja, Lingala, Sepedi."""
    syllables = []
    i = 0
    w = word.lower()

    while i < len(w):
        # Prenasalized onset
        onset = ""
        nucleus = ""
        coda = ""

        if i + 1 < len(w) and w[i:i+2] in prenasal_clusters:
            onset = w[i:i+2]
            i += 2
        else:
            while i < len(w) and w[i] not in vowels:
                onset += w[i]
                i += 1

        # Vowel
        while i < len(w) and w[i] in vowels:
            nucleus += w[i]
            i += 1

        # Optional nasal coda (rare)
        if i < len(w):
            if w[i] in ("m", "n"):
                coda = w[i]
                i += 1
            elif i + 1 == len(w) and w[i] == 'l':
                coda = w[i]
                i += 1

        syllables.append(onset + nucleus + coda)

    return syllables


# list of valid consonant clusters in Shona orthography
def syllabify_shona(word, clusters = ["ngw","ny","sh","ch","bh","dh","sv","zv","mh","nh"]):
    sylls = []
    i = 0
    n = len(word)
    while i < n:
        # vowel nucleus
        if word[i] in "aeiou":
            # this case: vowel alone
            sylls.append(word[i])
            i += 1
            continue

        # check 3-char clusters
        if i+2 < n and word[i:i+3] in clusters and i+3 < n and word[i+3] in "aeiou":
            sylls.append(word[i:i+4])  # cluster + vowel
            i += 4
            continue

        # check 2-char clusters
        if i+1 < n and word[i:i+2] in clusters and i+2 < n and word[i+2] in "aeiou":
            sylls.append(word[i:i+3])
            i += 3
            continue

        # single consonant + vowel
        if i+1 < n and word[i] not in "aeiou" and word[i+1] in "aeiou":
            sylls.append(word[i:i+2])
            i += 2
        else:
            # vowel or unexpected char alone
            sylls.append(word[i])
            i += 1
    return sylls


# vowels and diphthongs in Hausa
def syllabify_hausa(word, diphthongs = ["ai","au","iu","ui"], vowels = "aeiou"):
    word = word.lower().strip()
    syllables = []
    i = 0
    n = len(word)

    while i < n:
        # Onset
        onset = ""
        if i < n and word[i] not in vowels:
            onset = word[i]
            i += 1
        else:
            # treat as if glottal stop onset if vowel starts word
            onset = "ʼ"  # glottal stop marker
        # Nucleus: check diphthongs first
        nucleus = ""
        if i + 1 < n and word[i:i+2] in diphthongs:
            nucleus = word[i:i+2]
            i += 2
        elif i < n and word[i] in vowels:
            nucleus = word[i]
            i += 1
        else:
            # no vowel found (edge case)
            nucleus = ""
        # Coda: allow one consonant but no cluster
        coda = ""
        if i < n and word[i] not in vowels:
            # check next is consonant not followed by vowel
            if i + 1 < n and word[i+1] in vowels:
                # leave consonant for next onset
                coda = ""
            else:
                coda = word[i]
                i += 1

        syllables.append(onset + nucleus + coda)
    return syllables


def syllabify_northern_sotho(word):
    vowels = set("aeiou")
    nasals = {"m", "n"}

    w = word.lower()
    syllables = []
    i = 0
    n = len(w)

    while i < n:
        onset = ""

        # syllabic nasal before consonant
        if w[i] in nasals:
            if i + 1 < n and w[i + 1] not in vowels:
                syllables.append(w[i])
                i += 1
                continue

        # onset consonants
        while i < n and w[i] not in vowels:
            onset += w[i]
            i += 1

        # vowel nucleus (always exactly one)
        if i < n and w[i] in vowels:
            nucleus = w[i]
            i += 1
        else:
            nucleus = ""

        syllables.append(onset + nucleus)

    return syllables


LANG_RULES = {
    # Bantu languages
    "Ganda": syllabify_bantu,  # Luganda
    "Shona": syllabify_shona,  # Shona
    "Nyanja": syllabify_bantu,  # Nyanja
    "Lingala": syllabify_bantu,  # Lingala
    "Northern-Sotho": syllabify_northern_sotho, # Northern Sotho

    # Afro-Asiatic languages (Somali, Oromo, Hausa, Amharic)
    # They have more complex structures, but CV-heavy rules still give reasonable approximations.
    "Somali": universal_syllabify,
    "Oromo": universal_syllabify,
    "Hausa": syllabify_hausa,
    "Amharic": universal_syllabify,

    # Niger-Congo but not Bantu
    "Yoruba": universal_syllabify,  # Yoruba: tone sandhi makes segmentation hard
    "Igbo": universal_syllabify,  # Igbo: reasonably CV but has vowel harmony + tone
    "Fula": universal_syllabify,  # Fula
    "Wolof": syllabify_bantu,      # Wolof (close enough for CV rules)

    # Everything else -> universal syllabifier
}


# -----------------------------------------------------------
# Main public function for any FLEURS language
# -----------------------------------------------------------

def syllabify(language_code, text):
    """
    Syllabify text for *any* FLEURS language.
    Uses language-specific rules when available,
    otherwise falls back to universal maximal-onset syllabifier.
    """
    text = normalize_text(text)
    words = text.split()

    if language_code in LANG_RULES:
        fn = LANG_RULES[language_code]
    else:
        fn = universal_syllabify  # fallback

    out = []
    for w in words:
        if language_code == 'Shona':
            w = normalize_shona_prefixes(w)
        out.extend(fn(w))
    return out

parser = argparse.ArgumentParser()
parser.add_argument('--audio-dir', default='/data/sls/static/fleurs/train/Ganda')
parser.add_argument('--text-path', default='/data/sls/static/fleurs/train/Ganda')
parser.add_argument('--tsv-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/fleurs/Ganda_vad/train.tsv')
parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/fleurs/Ganda_vad/train_gradseg_style.json')
parser.add_argument('--bad-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/fleurs/Ganda_vad/train_bad_files.txt')
parser.add_argument('--lang')
parser.add_argument('--no-tone', type=ast.literal_eval, default='True')
args = parser.parse_args()
audio_dir = Path(args.audio_dir)
text_path = Path(args.text_path)

with open(args.tsv_path, 'r') as f:
    lines = f.read().strip().split('\n')
    root = lines.pop(0)

meta_dict = {}
vocab = []
for l in lines:
    fn = l.strip().split('\t')[0]
    fpath = Path(root) / fn 
    wav, sr = torchaudio.load(fpath)
    dur = wav.shape[-1]

    wav_id = Path(fn).stem
    text_fn = text_path / (wav_id + '.json')
    with open(text_fn, 'r') as f:
        text = json.load(f)['transcription']
        print(fn, text, dur)

        new_text = []
        for w in text.split():
            syls = syllabify(args.lang, w)
            for syl in syls:
                if syl not in vocab:
                    vocab.append(syl)
            new_text.extend(syls)
        text = deepcopy(new_text)
        if not len(text):
            print(f'{fn} without transcripts after postprocessing')
            print(fn, file=f_bad)
            continue
        else:
            print(fn, text, dur)

        time_step = dur // len(text)

        meta_dict[fn] = [
            [i*time_step, (i+1)*time_step, w]
            for i, w in enumerate(text)
        ]

with Path(args.out_path).open('w', encoding='utf-8', newline='\n') as f_out:
    json.dump(meta_dict, f_out, ensure_ascii=False, indent=2)

print('Vocab size:', len(vocab))

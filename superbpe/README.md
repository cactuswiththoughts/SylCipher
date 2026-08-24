# SuperBPE (BPE+ text units for SylCipher)

This directory vendors [SuperBPE](https://github.com/PythonNut/superbpe) ([arXiv:2503.13423](https://arxiv.org/abs/2503.13423)) plus helpers that map SuperBPE pieces onto **pyphen+** syllables (**BPE+**). Keep a separate conda env: the custom `tokenizers` fork conflicts with stock Hugging Face tokenizers.

**BPE+ pipeline**

1. Train a whitespace-pretokenized BPE (`python -m train_tokenizer`).
2. `match_tokens.py` — nearest-edit-distance map from those BPE pieces to a pyphen+ dictionary → `syllable_bpe.txt`.
3. `tokenize_superbpe.py` — SuperBPE tokenize with a one-vowel-cluster constraint.
4. `greedy_tokenize.py` — greedy longest-match with the matched vocab (this is the unit stream SylCipher uses).

```bash
source ../path.sh
conda activate "${SUPERBPE_ENV}"   # default: superbpe
cd superbpe

# LibriSpeech (vocab 17k). Set s/e inside the script: 0–1 train, 10 match, 20 tokenize, 30 greedy.
# Stage 10 needs ${MANIFEST_ROOT}/librispeech_clean_vad/syllable_pyphen_plus_dict_2048.txt
sbatch run_libri_clean_vad.sh

# MLS, one language (vocab sizes: de 9500, nl 10350, fr 12000, it 5000)
sbatch run_mls.slurm german
# or: bash run_mls.sh
```

Corpus text for tokenizer training should sit in `superbpe/librispeech_clean_vad/` (or override `SUPERBPE_CORPUS`). MLS stage 0 writes `superbpe/mls_<lang>_vad/` from `$DATA_ROOT/MLS/mls_<lang>`.

Setup for the SuperBPE env (Python 3.12 + Rust) is unchanged below.

---

# SuperBPE: Space Travel for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.13423-b31b1b.svg)](https://arxiv.org/pdf/2503.13423) [![website](https://img.shields.io/badge/Website-superbpe.github.io-C16C8A)](https://superbpe.github.io/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FFD21E)](https://huggingface.co/collections/UW/superbpe-67db2338062faa07c7473ffa)

This repository contains the tokenizer training code. Code for other aspects of the project (e.g. evals, model scaling, data processing, wandb, train configs) will be added soon!

## Setup
First, clone the project with:
```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git
```
We use a custom [fork](https://github.com/alisawuffles/tokenizers-superbpe) of [huggingface/tokenizers](https://github.com/huggingface/tokenizers) which conflicts with the original.
Because of this, we recommend *always installing this project in its own virtual environment.*

### Setup virtual environment

#### Using `conda`
```bash
conda create -n superbpe python=3.12 rust
conda activate superbpe
pip install -r requirements.txt
```

#### Using `venv`
You will need to [install rust](https://www.rust-lang.org/tools/install) and Python 3.12.
Then, you can do:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data download
Our tokenizer training data is available [here](https://huggingface.co/datasets/UW/olmo-mix-1124-subset-p99).
You can download it using [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (after logging into your HuggingFace account) using:
```
mkdir olmo-mix-1124-subset-p99
cd olmo-mix-1124-subset-p99
huggingface-cli download UW/olmo-mix-1124-subset-p99 --repo-type dataset --local-dir .
```

## Tokenizer training
Training a SuperBPE tokenizer involves two stages:

1. **Stage 1:** Learn subwords by enforcing whitespace pretokenization (equivalent to regular BPE training).

```bash
python -m train_tokenizer \
    --output_dir tokenizers/olmo2_bpe \
    --corpus_dir olmo-mix-1124-subset-p99/train \
    --num_bytes $((10**10)) \
    --vocab_size 200000 \
    --do_whitespace_pretokenization true
```

2. **Stage 2:** Learn superwords by resuming tokenizer training, but this time skip the whitespace pretokenization step.

```bash
orig_tokenizer_dir=tokenizers/olmo2_bpe
num_inherit_merges=180000
output_dir=tokenizers/olmo2_superbpe

mkdir -p $output_dir

# inherit the first num_inherit_merges from the BPE tokenizer
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt

# specifies the same training files used in stage 1
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size 200000 \
    --do_whitespace_pretokenization false
```

After tokenizer training, you need to update the `decoder` field in the `tokenizer.json` to make sure it looks like this.

```
"decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
}
```

## Citation 

If you found this codebase helpful, please cite

```
@inproceedings{liu-etal-2025-superbpe,
  title={{SuperBPE}: Space travel for language models},
  author={Alisa Liu and Jonathan Hayase and Valentin Hofmann and Sewoong Oh and Noah A Smith and Yejin Choi},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://arxiv.org/abs/2503.13423}
}
```

# pyphen+ (and related text units)

This folder holds **text-side** syllable tokenizers used by SylCipher. It is not a replacement for the [Sylber](https://github.com/Berkeley-Speech-Group/sylber) speech segmenter (`$SYLBER_ROOT`, default `../sylber`).

**pyphen** hyphenates English words with [Pyphen](https://pyphen.org/). **pyphen+** keeps that split, then repairs it with extra phonological rules (silent *-e* / *-ed*, VCCV vs VCV, merge onset clusters that have no vowel) using `syllapy` and `wordsegment`.

| Script | Unit |
|--------|------|
| `extract_libri_gt_syllable_boundary_pyphen.py` | pyphen, paired LibriSpeech (MFA TextGrids) |
| `extract_libri_gt_syllable_boundary_pyphen_extra_rules.py` | **pyphen+**, paired LibriSpeech |
| `extract_libri_gt_syllable_boundary_pyphen_plus_unpaired.py` | **pyphen+**, unpaired LM text |
| `extract_scoco_gt_syllable_boundary_pyphen_plus.py` | **pyphen+**, SpokenCOCO |
| `extract_libri_gt_syllable_boundary.py` | naive hyphenation (ablation) |
| `extract_units*.py` | Sylber / SyllableLM **speech** units (needs `syllablelm`) |

Install extras in the `fairseq` env:

```bash
pip install pyphen syllapy wordsegment textgrid
```

## LibriSpeech pyphen+ (paired)

MFA alignments live under `--align_dir` (one TextGrid per utterance). `--filelist` is a wav path list (same as the VAD manifest).

```bash
source ../path.sh
conda activate fairseq
cd sylber

for split in train valid valid_true; do
  python extract_libri_gt_syllable_boundary_pyphen_extra_rules.py \
    --filelist ${MANIFEST_ROOT}/librispeech_clean_vad/${split}_file_list.txt \
    --align_dir /path/to/LibriSpeech/alignments \
    --output_prefix ${MANIFEST_ROOT}/librispeech_clean_vad/${split}_syllable_pyphen_plus
done
```

Outputs (prefix `$output_prefix`): GradSeg-style JSON used by `prepare_text_sylber.sh` and a syllable dictionary (`syllable_pyphen_plus_dict_*.txt`) that BPE+ matching reads.

Plain pyphen (no extra rules): swap in `extract_libri_gt_syllable_boundary_pyphen.py`.

## Unpaired LM text (pyphen+)

```bash
python extract_libri_gt_syllable_boundary_pyphen_plus_unpaired.py \
  --input_file /path/to/lm.upper.lid.txt \
  --output_prefix ${MANIFEST_ROOT}/librispeech_clean_vad/train_librilm_syllable_pyphen_plus
```

Use the unpaired stream with `train_libri_sylber_bnd_semi_unpaired.slurm` / `data_curation/prepare_mixed_text.sh`.

## SpokenCOCO

```bash
python extract_scoco_gt_syllable_boundary_pyphen_plus.py \
  --filelist ${MANIFEST_ROOT}/spokencoco/without_silence/val2_file_list.txt \
  --align_path /path/to/SpokenCOCO_alignments.json \
  --output_prefix ${MANIFEST_ROOT}/spokencoco/without_silence/val2_syllable
```

## Speech units (Sylber / SyllableLM)

`extract_units_jstti_format.py` writes GradSeg pickles from a SyllableLM checkpoint. Point `--wav-dir`, `--file-list`, `--out-path`, and `--ckpt_prefix` at your data; this is independent of pyphen+.

Boundary F1 vs forced alignment: `eval_boundaries_gradseg_format.py`, `eval_segmentation.py`.

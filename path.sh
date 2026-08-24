#!/bin/bash
# Machine-specific paths. Edit this file or override the variables in your
# environment before running / submitting any script.
#
#   source /path/to/SylCipher/path.sh

if [ -n "${BASH_SOURCE[0]}" ]; then
  SYLCIPHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SYLCIPHER_ROOT="$(cd "$(dirname "$0")" && pwd)"
fi
export SYLCIPHER_ROOT

# fairseq 0.12.x checkout (must be on PYTHONPATH / have fairseq-train installed)
export FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-${HOME}/fairseq-0.12.1}"

# conda installation and the environment that has fairseq + PyTorch
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-fairseq}"

# Prepared manifests / features live outside this repo
export MANIFEST_ROOT="${MANIFEST_ROOT:-${SYLCIPHER_ROOT}/../manifest}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
export MODEL_ROOT="${MODEL_ROOT:-${HOME}/models}"

# Optional third-party tools
export KENLM_ROOT="${KENLM_ROOT:-${HOME}/softwares/kenlm/build/bin}"
export SYLBER_ROOT="${SYLBER_ROOT:-${SYLCIPHER_ROOT}/../sylber}"
export WAV2BOUNDARIES_ROOT="${WAV2BOUNDARIES_ROOT:-${SYLCIPHER_ROOT}/../wav2boundaries}"
export FAIRSEQ_W2B_ROOT="${FAIRSEQ_W2B_ROOT:-${HOME}/fairseq_fork}"
export SUPERBPE_ENV="${SUPERBPE_ENV:-superbpe}"

# fairseq user-dir implementations shipped in this repo
export SYLCIPHER_USER_DIR="${SYLCIPHER_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5/speecht5}"
export SYLCIPHER_UNISKIP_USER_DIR="${SYLCIPHER_UNISKIP_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_uniskip/speecht5}"
export SYLCIPHER_TL_USER_DIR="${SYLCIPHER_TL_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_transfer_learning/speecht5}"
export SYLCIPHER_TL_UNISKIP_USER_DIR="${SYLCIPHER_TL_UNISKIP_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_transfer_learning_uniskip/speecht5}"
export SYLCIPHER_MULTI_USER_DIR="${SYLCIPHER_MULTI_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_transfer_learning_multilingual/speecht5}"
export SYLCIPHER_MULTI_UNISKIP_USER_DIR="${SYLCIPHER_MULTI_UNISKIP_USER_DIR:-${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_transfer_learning_multilingual_uniskip/speecht5}"

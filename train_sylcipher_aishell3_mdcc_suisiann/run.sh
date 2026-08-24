#!/bin/bash
# Joint AISHELL-3 + MDCC + SuiSiann SylCipher.
# 1) sylber_bnd: combine per-language checkpoints, then train
# 2) transfer_learning: multilingual user-dir finetune
# 3) transfer_learning_uniskip: PUSM / unigram-skipgram stage
sbatch train_sylcipher_aishell3_mdcc_suisiann_sylber_bnd.slurm

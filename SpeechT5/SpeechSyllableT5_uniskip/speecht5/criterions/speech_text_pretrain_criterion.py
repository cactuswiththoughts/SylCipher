# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechTextPretrainCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    loss_weights: Optional[List[float]] = field(
        default_factory=lambda: [0.1,],
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    bart_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for cross entropy"},
    )
    masked_lm_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for cross entropy (masked lm at the encoder)"},
    )
    unigram_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for unigram l1 loss at the encoder"},
    )
    policy_pretrain_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for policy pretrain loss"},
    )
    policy_loss_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for policy gradient loss"},
    )
    word_freq_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for average word frequency"},
    )
    code_loss_weight: float = field(
        default=0.0,
        metadata={"help": "loss weight for dictionary and commitment loss of codebook of quantizer"},
    )    
    word_count_loss_weight: float = field(
        default=1000,
        metadata={"help": "loss weight for word count loss of segmenter"},
    )    
    text_loss_ratio: float = field(
        default=10,
        metadata={"help": "loss multiplier for text BART loss"},
    )    
    frame_target_loss_weight: float = field(
        default=0.1,
        metadata={"help": "loss weight for frame-level targets (from HuBERT clustering) of the segmenter"},
    )
    policy_pretrain_pos_weight_bce: float = field(
        default=1.5,
        metadata={"help": "pos_weight for pretraining the segmenter due to sparse boundary labels"},
    )
    boundary_sum_multiply: float = field(
        default=1.3,
        metadata={"help": "multiply boundary counts by this factor"},
    )
    mlm_unmasked_weight: float = field(
        default=0.0,
        metadata={"help": "weight for unmasked portion of MLM loss"},
    )

    

class SpeechTextPretrainCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, bart_weight, masked_lm_weight, unigram_weight, policy_pretrain_weight, policy_loss_weight, word_freq_weight, code_loss_weight, word_count_loss_weight, text_loss_ratio, frame_target_loss_weight, policy_pretrain_pos_weight_bce, boundary_sum_multiply, mlm_unmasked_weight, loss_weights=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss_weights = loss_weights
        self.bart_weight = bart_weight
        self.masked_lm_weight = masked_lm_weight
        self.unigram_weight = unigram_weight
        self.policy_pretrain_weight = policy_pretrain_weight
        self.policy_loss_weight = policy_loss_weight
        self.word_freq_loss_weight = word_freq_weight
        self.speech_padding_idx = task.dicts["audio"].pad()
        self.text_padding_idx = task.dicts["text"].pad()
        
        self.code_loss_weight = code_loss_weight
        self.word_count_loss_weight = word_count_loss_weight
        self.text_loss_ratio = text_loss_ratio
        self.frame_target_loss_weight = frame_target_loss_weight
        self.policy_pretrain_pos_weight_bce = policy_pretrain_pos_weight_bce
        self.boundary_sum_multiply = boundary_sum_multiply
        self.mlm_unmasked_weight = mlm_unmasked_weight
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        speech_net_outputs, text_net_output, speech_codebook_outs, text_codebook_out, speech_masked_lm_results, text_masked_lm_result, _, _, speech_meta_infos, loss_denses = model(**sample["net_input"])        

        speech_sample_sizes = []
        speech_mlm_losses = []
        speech_mlm_accuracies = []
        
        for speech_meta_info, speech_masked_lm_result in zip(speech_meta_infos, speech_masked_lm_results):
            speech_mlm_loss, speech_mlm_accuracy = self.compute_mlm_loss(model, speech_masked_lm_result, speech_meta_info, self.speech_padding_idx, reduce=reduce)
            speech_mlm_losses.append(speech_mlm_loss)
            
            speech_sample_size = (speech_meta_info["target"].size(0) if self.sentence_avg else speech_meta_info["ntokens"])
            speech_sample_sizes.append(speech_sample_size)
            speech_mlm_accuracies.append(speech_mlm_accuracy)
        
        text_mlm_loss, text_mlm_accuracy = self.compute_mlm_loss(model, text_masked_lm_result, sample["net_input"]["random_label"], self.text_padding_idx, reduce=reduce)
        
        text_sample_size = (
            sample["net_input"]["random_label"]["target"].size(0) if self.sentence_avg else sample["net_input"]["random_label"]["ntokens"]
        )

        speech_mlm_accuracy = speech_mlm_accuracies[0]
        loss = self.bart_weight * sum(speech_mlm_losses)
        
        loss += self.bart_weight  * self.text_loss_ratio * text_mlm_loss
        
        policy_logits_loss = None
        boundary_precision = None
        boundary_recall = None
        boundary_f1 = None
        boundary_accuracy = None
        
        aux_frame_clus = sample["net_input"]["aux_frame_clus"]
        frame_target_logits = speech_meta_infos[0]["frame_target_logits"]
        frame_target_preds = frame_target_logits.argmax(dim = -1)

        frame_target_preds_nomask = frame_target_preds[~speech_meta_infos[0]["policy_encoder_padding_mask"]]
        frame_target_nomask = aux_frame_clus[~speech_meta_infos[0]["policy_encoder_padding_mask"]]
        frame_target_accuracy = torch.sum(frame_target_preds_nomask == frame_target_nomask) / frame_target_nomask.size(0)
        
        frame_target_loss = self.frame_target_loss_weight * torch.nn.CrossEntropyLoss(reduction="sum")(frame_target_logits[~speech_meta_infos[0]["policy_encoder_padding_mask"],:], aux_frame_clus[~speech_meta_infos[0]["policy_encoder_padding_mask"]])
        
        # loss += frame_target_loss
        code_loss = None
        if "code_loss" in speech_meta_infos[0].keys():
            code_loss = speech_meta_infos[0]["code_loss"]
            loss += self.code_loss_weight * code_loss

        word_loss = None
        nonword_loss = None
        consecutive_word_loss = None

        boundary_padding_mask = speech_meta_infos[0]["policy_encoder_padding_mask"]
        num_frames = (~boundary_padding_mask).sum(1).detach()
        boundaries = sample["net_input"]["boundaries"]
        boundaries[boundary_padding_mask] = 0

        if "sum_after_mean_word_probs" in speech_meta_infos[0].keys():
            word_loss = F.l1_loss(speech_meta_infos[0]["sum_after_mean_word_probs"], boundaries.sum(1).float() * self.boundary_sum_multiply , reduction="sum")
            loss += self.word_count_loss_weight * word_loss

        if "sum_after_mean_nonword_probs" in speech_meta_infos[0].keys():
            nonword_loss = F.l1_loss(speech_meta_infos[0]["sum_after_mean_nonword_probs"],(num_frames - boundaries.sum(1) * self.boundary_sum_multiply).float(), reduction="sum")
            loss += self.word_count_loss_weight * nonword_loss

        if "sum_after_mean_consecutive_probs" in speech_meta_infos[0].keys():
            consecutive_word_loss = speech_meta_infos[0]["sum_after_mean_consecutive_probs"]
            loss += self.word_freq_loss_weight * consecutive_word_loss
        
        boundary_logits = speech_meta_infos[0]["policy_logits"].squeeze(-1)
        boundary_padding_mask = speech_meta_infos[0]["policy_encoder_padding_mask"]
        boundary_targets = sample["net_input"]["boundaries"]

        if self.policy_pretrain_weight > 0:
            bl = boundary_logits[~boundary_padding_mask].squeeze(-1)
            bt = boundary_targets[~boundary_padding_mask].squeeze(-1)
            pos_weight = bl.new_ones(bl.size()) * self.policy_pretrain_pos_weight_bce
            policy_logits_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight = pos_weight)(bl, bt.float())
            # loss += self.policy_pretrain_weight * policy_logits_loss
        
        boundary_preds = boundary_logits.new_zeros(boundary_logits.shape)
        boundary_preds[boundary_logits > 0] = 1
        boundary_preds[boundary_logits <= 0] = 0
        boundary_preds_nomask = boundary_preds[~boundary_padding_mask].squeeze()
        boundaries_nomask = boundary_targets[~boundary_padding_mask]
        
        tp = (boundaries_nomask * boundary_preds_nomask).sum()
        # tn = ((1 - boundaries_nomask) * (1 -boundary_preds_nomask)).sum()
        fp = ((1 - boundaries_nomask) * boundary_preds_nomask).sum()
        fn = (boundaries_nomask * (1 - boundary_preds_nomask)).sum()
        
        epsilon = 1e-7
        
        boundary_precision = tp / (tp + fp + epsilon)
        boundary_recall = tp / (tp + fn + epsilon)
        
        boundary_f1 = 2* (boundary_precision*boundary_recall) / (boundary_precision + boundary_recall + epsilon)
        
        boundary_accuracy = torch.sum(boundary_preds_nomask == boundaries_nomask) / boundary_preds_nomask.size(0)
        
        total_boundary_frames = (~boundary_padding_mask).sum()
        
        logging_output = {
            "loss": loss.item(),
            "ntokens": speech_meta_info["ntokens"],
            "nsentences": speech_meta_info["target"].size(0),
            "speech_mlm_loss": speech_mlm_loss.item(),
            "sample_size": speech_sample_size,
            "text_ntokens": sample["net_input"]["random_label"]["ntokens"],
            "text_nsentences": sample["net_input"]["random_label"]["target"].size(0),
            "text_mlm_loss": text_mlm_loss.item(),
            "text_sample_size": text_sample_size,
        }
        logging_output["frame_target_accuracy"] = frame_target_accuracy.item()
        logging_output["frame_target_loss"] = frame_target_loss.item()
        logging_output["speech_mlm_accuracy"] = speech_mlm_accuracy.item()
        logging_output["text_mlm_accuracy"] = text_mlm_accuracy.item()
        if "code_loss" in speech_meta_infos[0].keys():
            logging_output["code_loss"] = speech_meta_infos[0]["code_loss"].item()
        if policy_logits_loss is not None:
            logging_output["policy_logits_loss"] = self.policy_pretrain_weight * policy_logits_loss.item()
        logging_output["boundary_precision"] = boundary_precision.item()
        logging_output["boundary_recall"] = boundary_recall.item()
        logging_output["boundary_f1"] = boundary_f1.item()
        logging_output["boundary_accuracy"] = boundary_accuracy.item()
        logging_output["total_frames"] = total_boundary_frames.item()

        if word_loss is not None:
            logging_output["word_loss"] = word_loss.item()

        if nonword_loss is not None:
            logging_output["nonword_loss"] = nonword_loss.item()

        if consecutive_word_loss is not None:
            logging_output["consecutive_word_loss"] = consecutive_word_loss.item()
    
        if "prob_perplexity" in speech_codebook_outs[0]:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(speech_codebook_outs)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            if len(self.loss_weights) > len(extra_losses):
                modified_loss_weight = self.loss_weights[len(extra_losses):]
            else:
                modified_loss_weight = self.loss_weights

            # assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, modified_loss_weight):
                if coef != 0 and p is not None:
                    p = coef * p.float() * speech_sample_size
                    loss += p
                    logging_output[f"speech_loss_{n}"] = p.item()

        if 'speech_loss_prob_perplexity' in logging_output:
            code_perplexities = [speech_codebook_out['code_perplexity'].item() for speech_codebook_out in speech_codebook_outs]
            logging_output['speech_code_perplexity'] = sum(code_perplexities) / len(code_perplexities)
            
            
        if "prob_perplexity" in text_codebook_out:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(text_codebook_out)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            if len(self.loss_weights) > len(extra_losses):
                modified_loss_weight = self.loss_weights[len(extra_losses):]
            else:
                modified_loss_weight = self.loss_weights

            # assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, modified_loss_weight):
                if coef != 0 and p is not None:
                    p = coef * p.float() * text_sample_size
                    loss += p
                    logging_output[f"text_loss_{n}"] = p.item()

        if 'text_loss_prob_perplexity' in logging_output:
            logging_output['text_code_perplexity'] = text_codebook_out['code_perplexity'].item()
            
        logging_output['uniskip_loss'] = loss_denses[0]
        if loss_denses[0] == 0:
            loss = loss_denses[0]
        else:
            loss += loss_denses[0]
        return loss, speech_sample_size, logging_output
    
    
    def compute_mlm_loss(self, model, net_output, sample, padding_idx, reduce=True):
        # print(net_output[0].size())
        # print(sample["target"].size())
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        # print(lprobs.size(), target.size())
        # print(sample["net_input"]["src_tokens"][0])
        # print(sample["target"][0])
        # print(sample["net_input"]["src_tokens"].size(), target.size())
        lprobs_unmasked = lprobs[sample["net_input"]["src_tokens"].view(-1) == target]
        target_unmasked = target[sample["net_input"]["src_tokens"].view(-1) == target]
        unmasked_loss =  F.nll_loss(
            lprobs_unmasked,
            target_unmasked,
            ignore_index=padding_idx,
            reduction="sum" if reduce else "none",
        )
        lprobs = lprobs[sample["net_input"]["src_tokens"].view(-1) != target]
        target = target[sample["net_input"]["src_tokens"].view(-1) != target]
        # print(lprobs.size(), target.size())
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=padding_idx,
            reduction="sum" if reduce else "none",
        )
        lprobs_pred = lprobs.argmax(-1)[target != padding_idx].squeeze()
        target_pred = target[target != padding_idx].squeeze()
        overall_accuracy = torch.sum(lprobs_pred == target_pred) / len(target_pred)

        return loss+ self.mlm_unmasked_weight * unmasked_loss, overall_accuracy


    def compute_loss(self, model, net_output, sample, padding_idx, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        speech_ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        speech_sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        speech_bart_loss_sum = sum(log.get("speech_bart_loss", 0) for log in logging_outputs)
        speech_mlm_loss_sum = sum(log.get("speech_mlm_loss", 0) for log in logging_outputs)
        # speech_masked_lm_loss_sum = sum(log.get("speech_masked_lm_loss", 0) for log in logging_outputs)
        # speech_unigram_loss_sum = sum(log.get("speech_unigram_loss", 0) for log in logging_outputs)
        
        text_ntokens = sum(log.get("text_ntokens", 0) for log in logging_outputs)
        text_sample_size = sum(log.get("text_sample_size", 0) for log in logging_outputs)
        text_bart_loss_sum = sum(log.get("text_bart_loss", 0) for log in logging_outputs)
        text_mlm_loss_sum = sum(log.get("text_mlm_loss", 0) for log in logging_outputs)
        word_loss_sum = sum(log.get("word_loss", 0) for log in logging_outputs)
        nonword_loss_sum = sum(log.get("nonword_loss", 0) for log in logging_outputs)
        consecutive_word_loss_sum = sum(log.get("consecutive_word_loss", 0) for log in logging_outputs)
        # text_masked_lm_loss_sum = sum(log.get("text_masked_lm_loss", 0) for log in logging_outputs)
        # text_unigram_loss_sum = sum(log.get("text_unigram_loss", 0) for log in logging_outputs)

        policy_logits_loss_sum = sum(log.get("policy_logits_loss", 0) for log in logging_outputs)
        frame_target_loss_sum = sum(log.get("frame_target_loss", 0) for log in logging_outputs)
        code_loss_sum = sum(log.get("code_loss", 0) for log in logging_outputs)
        uniskip_loss_sum = sum(log.get("uniskip_loss", 0) for log in logging_outputs)
        # policy_gradient_loss_sum = sum(log.get("policy_gradient_loss", 0) for log in logging_outputs)
        # reward_loss_sum = sum(log.get("reward", 0) for log in logging_outputs)
        
        # word_freq_loss_sum = sum(log.get("word_freq_loss", 0) for log in logging_outputs)
        
        frame_sum = sum(log.get("total_frames", 0) for log in logging_outputs)

        boundary_precision = sum(log.get("boundary_precision", 0)  for log in logging_outputs) / len(logging_outputs)
        boundary_recall = sum(log.get("boundary_recall", 0)  for log in logging_outputs) / len(logging_outputs)
        boundary_f1 = sum(log.get("boundary_f1", 0) for log in logging_outputs) / len(logging_outputs)
        boundary_accuracy = sum(log.get("boundary_accuracy", 0)  for log in logging_outputs) / len(logging_outputs)
        
        frame_target_accuracy = sum(log.get("frame_target_accuracy", 0) for log in logging_outputs ) / len(logging_outputs)
        speech_mlm_accuracy = sum(log.get("speech_mlm_accuracy", 0) for log in logging_outputs ) / len(logging_outputs)
        text_mlm_accuracy = sum(log.get("text_mlm_accuracy", 0) for log in logging_outputs ) / len(logging_outputs)        
        

        

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / speech_sample_size / math.log(2), speech_sample_size, round=3
        )
        metrics.log_scalar(
            "bart_loss", speech_bart_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "speech_mlm_loss", speech_mlm_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        # metrics.log_scalar(
        #     "word_freq_loss", word_freq_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "speech_masked_lm_loss", speech_masked_lm_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "speech_unigram_loss", speech_unigram_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        # )
        if speech_sample_size != speech_ntokens:
            metrics.log_scalar(
                "speech_nll_loss", speech_bart_loss_sum / speech_ntokens / math.log(2), speech_ntokens, round=3
            )
            metrics.log_derived(
                "speech_ppl", lambda meters: utils.get_perplexity(meters["speech_nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "speech_ppl", lambda meters: utils.get_perplexity(meters["speech_bart_loss"].avg)
            )

        if "speech_loss_prob_perplexity" in logging_outputs[0].keys():
            val = sum(log["speech_loss_prob_perplexity"] for log in logging_outputs)
            metrics.log_scalar("speech_loss_prob_perplexity", val / speech_sample_size / math.log(2), round=3)
        if "speech_code_perplexity" in logging_outputs[0].keys():
            val = sum(log["speech_code_perplexity"] for log in logging_outputs)
            metrics.log_scalar("speech_code_perplexity", val / len(logging_outputs), round=3)
            
            
        metrics.log_scalar(
            "text_bart_loss", text_bart_loss_sum / text_sample_size / math.log(2), text_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "text_mlm_loss", text_mlm_loss_sum / text_sample_size / math.log(2), text_ntokens, 2, round=3
        )
        # metrics.log_scalar(
        #     "text_masked_lm_loss", text_masked_lm_loss_sum / text_sample_size / math.log(2), text_ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "text_unigram_loss", text_unigram_loss_sum / text_sample_size / math.log(2), text_ntokens, 2, round=3
        # )
        if text_sample_size != text_ntokens:
            metrics.log_scalar(
                "text_nll_loss", text_bart_loss_sum / text_ntokens / math.log(2), text_ntokens, round=3
            )
            metrics.log_derived(
                "text_ppl", lambda meters: utils.get_perplexity(meters["text_nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "text_ppl", lambda meters: utils.get_perplexity(meters["text_bart_loss"].avg)
            )

        if "text_loss_prob_perplexity" in logging_outputs[0].keys():
            val = sum(log["text_loss_prob_perplexity"] for log in logging_outputs)
            metrics.log_scalar("text_loss_prob_perplexity", val / text_sample_size / math.log(2), round=3)
        if "text_code_perplexity" in logging_outputs[0].keys():
            val = sum(log["text_code_perplexity"] for log in logging_outputs)
            metrics.log_scalar("text_code_perplexity", val / len(logging_outputs), round=3)

        metrics.log_scalar(
            "policy_logits_loss", policy_logits_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "frame_target_loss", frame_target_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "code_loss", code_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "uniskip_loss", uniskip_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "word_loss", word_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "nonword_loss", nonword_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "consecutive_word_loss", consecutive_word_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        )
        metrics.log_scalar(
            "boundary_precision", boundary_precision, 1, 2, round=3
        )
        metrics.log_scalar(
            "boundary_recall", boundary_recall, 1, 2, round=3
        )
        metrics.log_scalar(
            "boundary_f1", boundary_f1, 1, 2, round=3
        )
        metrics.log_scalar(
            "boundary_accuracy", boundary_accuracy, 1, 2, round=3
        )
        metrics.log_scalar(
            "frame_target_accuracy", frame_target_accuracy, 1, 2, round=3
        )
        metrics.log_scalar(
            "speech_mlm_accuracy", speech_mlm_accuracy, 1, 2, round=3
        )
        metrics.log_scalar(
            "text_mlm_accuracy", text_mlm_accuracy, 1, 2, round=3
        )
        # metrics.log_scalar(
        #     "policy_gradient_loss", policy_gradient_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "reward", reward_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
        # )
            
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

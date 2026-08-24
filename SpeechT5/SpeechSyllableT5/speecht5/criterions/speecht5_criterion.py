# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import re
from dataclasses import dataclass

import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from speecht5.criterions.text_to_speech_loss import TexttoSpeechLoss
from speecht5.criterions.text_pretrain_criterion import TextPretrainCriterion, TextPretrainCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from speecht5.criterions.speech_pretrain_criterion import SpeechPretrainCriterion, SpeechPretrainCriterionConfig
from speecht5.criterions.speech_to_text_loss import SpeechtoTextLoss, SpeechtoTextLossConfig             
from speecht5.criterions.speech_text_pretrain_gan import GANPretrainCriterion, GANPretrainCriterionConfig
from speecht5.criterions.speech_text_pretrain_criterion import SpeechTextPretrainCriterion, SpeechTextPretrainCriterionConfig                                 
from fairseq.logging.meters import safe_round

@dataclass
class SpeechT5CriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig, 
    TextPretrainCriterionConfig,
    SpeechPretrainCriterionConfig,
    SpeechtoTextLossConfig,
    GANPretrainCriterionConfig,
    SpeechTextPretrainCriterionConfig
    ):
    pass

@register_criterion(
    "speecht5", dataclass=SpeechT5CriterionConfig
)
class SpeechT5Criterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        pred_masked_weight, 
        pred_nomask_weight,
        gan_loss_weight,
        gradient_penalty,
        entropy_penalty,
        entropy_threshold,
        loss_weights=None, 
        log_keys=None,
        ignore_prefix_size=0,
        report_accuracy=False,
        report_similarity=False,
        similarity_meanpool=False,
        use_masking=True,
        use_weighted_masking=False,
        loss_type="L1",
        bce_pos_weight=5.0,
        bce_loss_lambda=1.0,
        use_guided_attn_loss=False,
        num_heads_applied_guided_attn=2,
        ce_weight=1.0,
        ctc_weight=0.0,
        hubert_weight=1.0,
        dec_weight=1.0,
        bart_weight=1.0,
        masked_lm_weight=1.0,
        unigram_weight=1.0,
        policy_pretrain_weight=1.0,
        policy_loss_weight=1.0,
        word_freq_weight=1.0,
        code_loss_weight = 0,
        word_count_loss_weight = 1000,
        text_loss_ratio = 10,
        frame_target_loss_weight = 0.0,
        policy_pretrain_pos_weight_bce = 1.5,
        boundary_sum_multiply = 1.3,
        mlm_unmasked_weight = 0.0,
        smoothing = 0.0,
        smoothing_one_sided = False,
        probabilistic_grad_penalty_slicing = False 
    ):
        super().__init__(task)
        self.speech_criterion = TexttoSpeechLoss(
            task,
            sentence_avg,
            use_masking,
            use_weighted_masking,
            loss_type,
            bce_pos_weight,
            bce_loss_lambda,
            use_guided_attn_loss,
            num_heads_applied_guided_attn=num_heads_applied_guided_attn,
        )
        self.text_criterion = SpeechtoTextLoss(
            SpeechtoTextLossConfig,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            report_similarity,
            similarity_meanpool,
            ce_weight,
            ctc_weight
        )
        self.speech_text_pretrain_criterion = SpeechTextPretrainCriterion(
            task,
            sentence_avg,
            bart_weight,
            masked_lm_weight,
            unigram_weight,
            policy_pretrain_weight,
            policy_loss_weight,
            word_freq_weight,
            code_loss_weight,
            word_count_loss_weight,
            text_loss_ratio,
            frame_target_loss_weight,
            policy_pretrain_pos_weight_bce,
            boundary_sum_multiply,
            mlm_unmasked_weight,
            loss_weights,
        )
        self.policy_pretrain_weight = policy_pretrain_weight
        self.policy_loss_weight = policy_loss_weight


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        task_name = sample['task_name']
        # print(task_name, getattr(sample["net_input"], "task_name", None,  getattr(sample["net_input"]["random_label"], "task_name", None)))
        # print(sample["net_input"].keys())
        if task_name == 's2t' or task_name == 's2c':
            if self.policy_loss_weight > 0:
                sample["net_input"]["speech_prenet_mode"] = "policy_loss"
            else:
                sample["net_input"]["speech_prenet_mode"] = "policy_pretrain"
            return self.text_criterion(model, sample, reduce)
        elif task_name == 't2s' or task_name == 's2s':
            return self.speech_criterion(model, sample)
        elif 'pretrain' in task_name:
            # if model.speech_pretrain_step(model.update_num):
            #     # print('speech_pretrain', model.update_num)
            #     sample['task_name'] = "speech_pretrain"
            #     sample["net_input"]["task_name"] = "speech_pretrain"
            #     sample["net_input"]["random_label"]["task_name"] = "speech_pretrain"
            #     sample["net_input"]["random_label"]["net_input"]["task_name"] = "speech_pretrain"
            #     sample["net_input"]["random_src_tokens"] = [sample["target"], sample["net_input"]["random_label"]["target"]]
            #     return self.speech_pretrain_criterion(model, sample, reduce)
            # else:
            #     # print('text_pretrain', model.update_num)
            #     sample['task_name'] = "text_pretrain"
            #     sample["net_input"]["task_name"] = "text_pretrain"
            #     sample["net_input"]["random_label"]["task_name"] = "text_pretrain"
            #     sample["net_input"]["random_label"]["net_input"]["task_name"] = "text_pretrain"
            #     sample["net_input"]["random_label"]["net_input"]["random_src_tokens"] = [sample["net_input"]["random_label"]["target"], sample["target"]]
            #     return self.text_pretrain_criterion(model, sample["net_input"]["random_label"], reduce)
            sample["task_name"] = "speech_text_pretrain"
            sample["net_input"]["task_name"] = "speech_text_pretrain"
            sample["net_input"]["random_label"]["task_name"] = "speech_text_pretrain"
            sample["net_input"]["random_label"]["net_input"]["task_name"] = "speech_text_pretrain"
            if self.policy_loss_weight > 0:
                sample["net_input"]["speech_prenet_mode"] = "policy_loss"
            else:
                sample["net_input"]["speech_prenet_mode"] = "policy_pretrain"
            sample["net_input"]["random_src_tokens"] = sample["net_input"]["random_label"]["target"]
            return self.speech_text_pretrain_criterion(model, sample, reduce)
        else:
            print('nah where are you?')
                

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        logging_outputs_dict = {}
        for logging_output in logging_outputs:
            for task_name in logging_output:
                if task_name not in ['s2t', 't2s', 's2c', 's2s', 'text_pretrain', 'speech_pretrain', 'speech_text_pretrain','speech_text_pretrain_gan']:
                    continue

                if task_name not in logging_outputs_dict:
                    logging_outputs_dict[task_name] = []
                logging_outputs_dict[task_name].append(logging_output[task_name])

        for task_name in logging_outputs_dict:
            # print(task_name)
            if task_name == 's2t':
                # LabelSmoothedCrossEntropyCriterion.reduce_metrics([logging_output['s2t'] for logging_output in logging_outputs])
                s2t_logging_output = logging_outputs_dict[task_name]
                # s2t_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
                loss_sum = sum(log.get("loss", 0) for log in s2t_logging_output)
                nll_loss_sum = sum(log.get("nll_loss", 0) for log in s2t_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in s2t_logging_output)
                ce_loss_sum = sum(log.get("ce_loss", 0) for log in s2t_logging_output)
                ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in s2t_logging_output)

                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2t_logging_output))
                metrics.log_scalar(
                    "s2t_loss", loss_sum / sample_size / math.log(2), sample_size, 1, round=3
                )

                metrics.log_scalar(
                    "s2t_nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, 2, round=3
                )
                metrics.log_derived(
                    "s2t_ppl", lambda meters: utils.get_perplexity(meters["s2t_nll_loss"].avg, 2)
                )
                metrics.log_scalar(
                    "ctc_loss", ctc_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "ce_loss", ce_loss_sum / ntokens, ntokens, 2, round=3
                )

                total = utils.item(sum(log.get("total", 0) for log in s2t_logging_output))
                if total > 0:
                    metrics.log_scalar("s2t_total", total)
                    n_correct = utils.item(
                        sum(log.get("n_correct", 0) for log in s2t_logging_output)
                    )
                    metrics.log_scalar("s2t_n_correct", n_correct)
                    metrics.log_derived(
                        "s2t_accuracy",
                        lambda meters: round(
                            meters["s2t_n_correct"].sum * 100.0 / meters["s2t_total"].sum, 3
                        )
                        if meters["s2t_total"].sum > 0
                        else float("nan"),
                        2
                    )
                cos_sim_total = utils.item(sum(log.get("cosine_similarity_total", 0) for log in s2t_logging_output))
                # print(cos_sim_total)
                if cos_sim_total != 0:
                    metrics.log_scalar("s2t_cosine_similarity_total", cos_sim_total)
                    cosine_similarity_sum = utils.item(
                        sum(log.get("cosine_similarity_sum", 0) for log in s2t_logging_output)
                    )
                    metrics.log_scalar("s2t_cosine_similarity_sum", cosine_similarity_sum)
                    metrics.log_derived(
                        "s2t_cosine_similarity",
                        lambda meters: round(
                            meters["s2t_cosine_similarity_sum"].sum / meters["s2t_cosine_similarity_total"].sum, 3
                        )
                        if meters["s2t_cosine_similarity_total"].sum > 0
                        else float("nan"),
                        2
                    )
                c_errors = sum(log.get("c_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_c_errors", c_errors)
                c_total = sum(log.get("c_total", 0) for log in s2t_logging_output)
                metrics.log_scalar("_c_total", c_total)
                w_errors = sum(log.get("w_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_w_errors", w_errors)
                wv_errors = sum(log.get("wv_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_wv_errors", wv_errors)
                w_total = sum(log.get("w_total", 0) for log in s2t_logging_output)
                metrics.log_scalar("_w_total", w_total)
                if c_total > 0:
                    metrics.log_derived(
                        "uer",
                        lambda meters: safe_round(
                            meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                        )
                        if meters["_c_total"].sum > 0
                        else float("nan"),
                    )
                if w_total > 0:
                    metrics.log_derived(
                        "wer",
                        lambda meters: safe_round(
                            meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                        )
                        if meters["_w_total"].sum > 0
                        else float("nan"),
                    )
                    metrics.log_derived(
                        "raw_wer",
                        lambda meters: safe_round(
                            meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                        )
                        if meters["_w_total"].sum > 0
                        else float("nan"),
                    )

            if task_name == 't2s':
                # TTSLossCriterion.reduce_metrics([logging_output['t2s'] for logging_output in logging_outputs])
                # t2s_sum = sum(log.get("speech_loss", 0) for log in logging_outputs)
                t2s_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in t2s_logging_output)
                l1_loss_sum = sum(log.get("l1_loss", 0) for log in t2s_logging_output)
                l2_loss_sum = sum(log.get("l2_loss", 0) for log in t2s_logging_output)
                bce_loss_sum = sum(log.get("bce_loss", 0) for log in t2s_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in t2s_logging_output))
                metrics.log_scalar(
                    "t2s_loss", loss_sum / sample_size, sample_size, 1, round=5
                )
                encoder_alpha_sum = sum(log.get("encoder_alpha", 0) for log in t2s_logging_output)
                decoder_alpha_sum = sum(log.get("decoder_alpha", 0) for log in t2s_logging_output)
                ngpu = sum(log.get("ngpu", 0) for log in t2s_logging_output)

                metrics.log_scalar(
                    "t2s_l1_loss", l1_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_l2_loss", l2_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_bce_loss", bce_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_encoder_alpha", encoder_alpha_sum / sample_size, sample_size, round=5
                )
                metrics.log_scalar(
                    "t2s_decoder_alpha", decoder_alpha_sum / sample_size, sample_size, round=5
                )

                if "enc_dec_attn_loss" in t2s_logging_output[0]:
                    enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in t2s_logging_output)
                    metrics.log_scalar(
                        "t2s_enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
                    )

            if task_name == 's2c':
                s2c_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in s2c_logging_output)
                nll_loss_sum = sum(log.get("nll_loss", 0) for log in s2c_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in s2c_logging_output)

                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2c_logging_output))
                metrics.log_scalar(
                    "s2c_loss", loss_sum / sample_size / math.log(2), sample_size, 1, round=3
                )

                metrics.log_scalar(
                    "s2c_nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, 2, round=3
                )

                total = utils.item(sum(log.get("total", 0) for log in s2c_logging_output)) 
                if total > 0:
                    metrics.log_scalar("s2c_total", total)
                    n_correct = utils.item(sum(log.get("n_correct", 0) for log in s2c_logging_output))
                    metrics.log_scalar("s2c_n_correct", n_correct)
                    metrics.log_derived(
                        "s2c_accuracy",
                        lambda meters: round(
                            meters["s2c_n_correct"].sum * 100.0 / meters["s2c_total"].sum, 3
                        )
                        if meters["s2c_total"].sum > 0
                        else float("nan"),
                        2
                    )
            
            if task_name == 's2s':
                s2s_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in s2s_logging_output)
                l1_loss_sum = sum(log.get("l1_loss", 0) for log in s2s_logging_output)
                l2_loss_sum = sum(log.get("l2_loss", 0) for log in s2s_logging_output)
                bce_loss_sum = sum(log.get("bce_loss", 0) for log in s2s_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2s_logging_output))
                metrics.log_scalar(
                    "s2s_loss", loss_sum / sample_size, sample_size, 1, round=5
                )
                encoder_alpha_sum = sum(log.get("encoder_alpha", 0) for log in s2s_logging_output)
                decoder_alpha_sum = sum(log.get("decoder_alpha", 0) for log in s2s_logging_output)
                ngpu = sum(log.get("ngpu", 0) for log in s2s_logging_output)

                metrics.log_scalar(
                    "s2s_l1_loss", l1_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_l2_loss", l2_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_bce_loss", bce_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_decoder_alpha", decoder_alpha_sum / sample_size, sample_size, round=5
                )

                if "enc_dec_attn_loss" in s2s_logging_output[0]:
                    enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in s2s_logging_output)
                    metrics.log_scalar(
                        "s2s_enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
                    )
                    
                    
            if task_name == "speech_text_pretrain_gan":
                gan_logging_output = logging_outputs_dict[task_name]
                generator_loss_sum = sum(log.get("generator_loss", 0) for log in gan_logging_output)
                generator_sample_size = max(1, sum(log.get("generator_sample_size", 0) for log in gan_logging_output))
                generator_dense_loss_sum = sum(log.get("generator_loss_dense", 0) for log in gan_logging_output)
                generator_token_loss_sum = sum(log.get("generator_loss_token", 0) for log in gan_logging_output)
                generator_entropy_loss_sum = sum(log.get("generator_loss_entropy", 0) for log in gan_logging_output)
                
                
                discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in gan_logging_output)
                discriminator_sample_size = max(1, sum(log.get("discriminator_sample_size", 0) for log in gan_logging_output))
                discriminator_dense_loss_sum = sum(log.get("discriminator_loss_dense", 0) for log in gan_logging_output)
                discriminator_token_loss_sum = sum(log.get("discriminator_loss_token", 0) for log in gan_logging_output)
                
                
                gp_loss_sum = sum(log.get("grad_pen", 0) for log in gan_logging_output)
                
                metrics.log_scalar(
                    "generator_loss", generator_loss_sum / generator_sample_size / math.log(2), generator_sample_size, round=3
                )
                metrics.log_scalar(
                    "generator_loss_dense", generator_dense_loss_sum / generator_sample_size / math.log(2), generator_sample_size, round=3
                )
                metrics.log_scalar(
                    "generator_loss_token", generator_token_loss_sum / generator_sample_size / math.log(2), generator_sample_size, round=3
                )
                metrics.log_scalar(
                    "generator_loss_entropy", generator_entropy_loss_sum / generator_sample_size / math.log(2), generator_sample_size, round=3
                )
                
                metrics.log_scalar(
                    "discriminator_loss", discriminator_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
                )
                metrics.log_scalar(
                    "discriminator_loss_dense", discriminator_dense_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
                )
                metrics.log_scalar(
                    "discriminator_loss_token", discriminator_token_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
                )
                
                metrics.log_scalar(
                    "grad_pen", gp_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
                )
                
                
            if task_name == 'speech_text_pretrain':
                
                speech_text_logging_outputs = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in speech_text_logging_outputs)
                speech_ntokens = sum(log.get("ntokens", 0) for log in speech_text_logging_outputs)
                speech_sample_size = sum(log.get("sample_size", 0) for log in speech_text_logging_outputs)
                speech_bart_loss_sum = sum(log.get("speech_bart_loss", 0) for log in speech_text_logging_outputs)
                speech_mlm_loss_sum = sum(log.get("speech_mlm_loss", 0) for log in speech_text_logging_outputs)
                # speech_masked_lm_loss_sum = sum(log.get("speech_masked_lm_loss", 0) for log in speech_text_logging_outputs)
                # speech_unigram_loss_sum = sum(log.get("speech_unigram_loss", 0) for log in speech_text_logging_outputs)
                
                text_ntokens = sum(log.get("text_ntokens", 0) for log in speech_text_logging_outputs)
                text_sample_size = sum(log.get("text_sample_size", 0) for log in speech_text_logging_outputs)
                text_bart_loss_sum = sum(log.get("text_bart_loss", 0) for log in speech_text_logging_outputs)
                text_mlm_loss_sum = sum(log.get("text_mlm_loss", 0) for log in speech_text_logging_outputs)
                # text_masked_lm_loss_sum = sum(log.get("text_masked_lm_loss", 0) for log in speech_text_logging_outputs)
                # text_unigram_loss_sum = sum(log.get("text_unigram_loss", 0) for log in speech_text_logging_outputs)
                
                policy_logits_loss_sum = sum(log.get("policy_logits_loss", 0) for log in speech_text_logging_outputs)
                frame_target_loss_sum = sum(log.get("frame_target_loss", 0) for log in speech_text_logging_outputs)
                code_loss_sum = sum(log.get("code_loss", 0) for log in speech_text_logging_outputs)
                # policy_gradient_loss_sum = sum(log.get("policy_gradient_loss", 0) for log in speech_text_logging_outputs)
                
                # reward_loss_sum = sum(log.get("reward", 0) for log in speech_text_logging_outputs)
                
                # word_freq_loss_sum = sum(log.get("word_freq_loss", 0) for log in speech_text_logging_outputs)
                        
                frame_sum = sum(log.get("total_frames", 0) for log in speech_text_logging_outputs)

                boundary_precision = sum(log.get("boundary_precision", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                boundary_recall = sum(log.get("boundary_recall", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                boundary_f1 = sum(log.get("boundary_f1", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                boundary_accuracy = sum(log.get("boundary_accuracy", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                
                frame_target_accuracy = sum(log.get("frame_target_accuracy", 0)  for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                speech_mlm_accuracy = sum(log.get("speech_mlm_accuracy", 0)  for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                text_mlm_accuracy = sum(log.get("text_mlm_accuracy", 0)  for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)

                # boundary_precision = sum(log.get("boundary_precision", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                # boundary_recall = sum(log.get("boundary_recall", 0) for log in speech_text_logging_outputs)  / len(speech_text_logging_outputs)
                # boundary_f1 = sum(log.get("boundary_f1", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                # boundary_accuracy = sum(log.get("boundary_accuracy", 0) for log in speech_text_logging_outputs) / len(speech_text_logging_outputs)
                
                word_loss_sum = sum(log.get("word_loss", 0) for log in speech_text_logging_outputs)
                nonword_loss_sum = sum(log.get("nonword_loss", 0) for log in speech_text_logging_outputs)
                consecutive_word_loss_sum = sum(log.get("consecutive_word_loss", 0) for log in speech_text_logging_outputs)
                

                # we divide by log(2) to convert the loss from base e to base 2
                metrics.log_scalar(
                    "loss", loss_sum / speech_sample_size / math.log(2), speech_sample_size, round=3
                )
                # metrics.log_scalar(
                #     "word_freq_loss", word_freq_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
                # )
                metrics.log_scalar(
                    "speech_bart_loss", speech_bart_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "speech_mlm_loss", speech_mlm_loss_sum / speech_sample_size / math.log(2), speech_ntokens, 2, round=3
                )
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

                if "speech_loss_prob_perplexity" in speech_text_logging_outputs[0].keys():
                    val = sum(log["speech_loss_prob_perplexity"] for log in speech_text_logging_outputs)
                    metrics.log_scalar("speech_loss_prob_perplexity", val / speech_sample_size / math.log(2), round=3)
                if "speech_code_perplexity" in speech_text_logging_outputs[0].keys():
                    val = sum(log["speech_code_perplexity"] for log in speech_text_logging_outputs)
                    metrics.log_scalar("speech_code_perplexity", val / len(speech_text_logging_outputs), round=3)
                    
                    
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

                if "text_loss_prob_perplexity" in speech_text_logging_outputs[0].keys():
                    val = sum(log["text_loss_prob_perplexity"] for log in speech_text_logging_outputs)
                    metrics.log_scalar("text_loss_prob_perplexity", val / text_sample_size / math.log(2), round=3)
                if "text_code_perplexity" in speech_text_logging_outputs[0].keys():
                    val = sum(log["text_code_perplexity"] for log in speech_text_logging_outputs)
                    metrics.log_scalar("text_code_perplexity", val / len(speech_text_logging_outputs), round=3)
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
            if task_name == 'text_pretrain':
                bart_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in bart_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in bart_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in bart_logging_output))
                bart_loss_sum = sum(log.get("bart_loss", 0) for log in bart_logging_output)
                masked_lm_loss_sum = sum(log.get("masked_lm_loss", 0) for log in bart_logging_output)
                unigram_loss_sum = sum(log.get("unigram_loss", 0) for log in bart_logging_output)

                # we divide by log(2) to convert the loss from base e to base 2
                metrics.log_scalar(
                    "text_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
                )
                metrics.log_scalar(
                    "bart_loss", bart_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "masked_lm_loss", masked_lm_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "unigram_loss", unigram_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                if sample_size != ntokens:
                    metrics.log_scalar(
                        "bart_nll_loss", bart_loss_sum / ntokens / math.log(2), ntokens, round=3
                    )
                    metrics.log_derived(
                        "bart_ppl", lambda meters: utils.get_perplexity(meters["bart_nll_loss"].avg)
                    )
                else:
                    metrics.log_derived(
                        "bart_ppl", lambda meters: utils.get_perplexity(meters["bart_loss"].avg)
                    )
                metrics.log_scalar("bart_wpb", ntokens, priority=180, round=1)

                val_prob_perplexity = 0
                val_code_perplexity = 0
                sample_size_pp = 0
                count_log_cp = 0
                for log in bart_logging_output:
                    if "loss_prob_perplexity" in log:
                        val_prob_perplexity = val_prob_perplexity + log["loss_prob_perplexity"]
                        sample_size_pp = sample_size_pp + log["sample_size"]
                    if "code_perplexity" in log:
                        val_code_perplexity = val_code_perplexity + log["code_perplexity"]
                        count_log_cp = count_log_cp + 1
                if val_prob_perplexity > 0:
                    metrics.log_scalar("text_loss_prob_perplexity", val_prob_perplexity / sample_size_pp / math.log(2), round=3)
                if val_code_perplexity > 0:
                    metrics.log_scalar("text_code_perplexity", val_code_perplexity / count_log_cp, round=3)

            if task_name == 'speech_pretrain':
                bart_logging_output = logging_outputs_dict[task_name]
                # print(bart_logging_output)
                loss_sum = sum(log.get("loss", 0) for log in bart_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in bart_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in bart_logging_output))
                bart_loss_sum = sum(log.get("bart_loss", 0) for log in bart_logging_output)
                masked_lm_loss_sum = sum(log.get("masked_lm_loss", 0) for log in bart_logging_output)
                unigram_loss_sum = sum(log.get("unigram_loss", 0) for log in bart_logging_output)
                                
                # we divide by log(2) to convert the loss from base e to base 2
                metrics.log_scalar(
                    "speech_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
                )
                metrics.log_scalar(
                    "speech_bart_loss", bart_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "speech_masked_lm_loss", masked_lm_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "speech_unigram_loss", unigram_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                if sample_size != ntokens:
                    metrics.log_scalar(
                        "speech_bart_nll_loss", bart_loss_sum / ntokens / math.log(2), ntokens, round=3
                    )
                    metrics.log_derived(
                        "speech_bart_ppl", lambda meters: utils.get_perplexity(meters["speech_bart_nll_loss"].avg)
                    )
                else:
                    metrics.log_derived(
                        "speech_bart_ppl", lambda meters: utils.get_perplexity(meters["speech_bart_loss"].avg)
                    )
                metrics.log_scalar("speech_bart_wpb", ntokens, priority=180, round=1)

                val_prob_perplexity = 0
                val_code_perplexity = 0
                sample_size_pp = 0
                count_log_cp = 0
                for log in bart_logging_output:
                    if "loss_prob_perplexity" in log:
                        val_prob_perplexity = val_prob_perplexity + log["loss_prob_perplexity"]
                        sample_size_pp = sample_size_pp + log["sample_size"]
                    if "code_perplexity" in log:
                        val_code_perplexity = val_code_perplexity + log["code_perplexity"]
                        count_log_cp = count_log_cp + 1
                if val_prob_perplexity > 0:
                    metrics.log_scalar("speech_loss_prob_perplexity", val_prob_perplexity / sample_size_pp / math.log(2), round=3)
                if val_code_perplexity > 0:
                    metrics.log_scalar("speech_code_perplexity", val_code_perplexity / count_log_cp, round=3)

        loss = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = max(1, sum(log.get("sample_size", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss", loss / sample_size, sample_size, 1, round=5
        )
        
        
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

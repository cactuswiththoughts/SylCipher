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
import numpy as np
from torch import autograd

@dataclass
class GANPretrainCriterionConfig(FairseqDataclass):
    gan_loss_weight: float = field(
        default=0.1,
        metadata={"help": "GAN loss weight"},
    )
    gradient_penalty: float = field(
        default=1.0,
        metadata={"help": "gradient penalty loss weight"},
    )
    entropy_penalty: float = field(
        default=1.0,
        metadata={"help": "entropy penalty loss weight"},
    )
    entropy_threshold: float = field(
        default=1.0,
        metadata={"help": "entropy penalty loss threshold after multiplying by weight"},
    )
    smoothing: float = field(
        default=0.0,
        metadata={"help": "smoothing for GAN labels"},
    )
    smoothing_one_sided: bool = field(
        default=False,
        metadata={"help": "if true, the fake label are not smoothed"},
    )
    probabilistic_grad_penalty_slicing: bool = field(
        default=False,
        metadata={"help": "if true, slice part of the data in the batch and temporal dimension for calculating gradient penalty"},
    )
    


class GANPretrainCriterion(FairseqCriterion):
    def __init__(self, task, gan_loss_weight, gradient_penalty, entropy_penalty, entropy_threshold, smoothing=0.0, smoothing_one_sided = False, probabilistic_grad_penalty_slicing= False):
        super().__init__(task)
        self.gan_loss_weight = gan_loss_weight
        self.gradient_penalty = gradient_penalty
        self.entropy_penalty = entropy_penalty
        self.entropy_threshold = entropy_threshold
        self.smoothing = smoothing
        self.smoothing_one_sided = smoothing_one_sided
        self.probabilistic_grad_penalty_slicing = probabilistic_grad_penalty_slicing
        
        
    def calc_gradient_penalty(self, model, real_data, fake_data):

        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        if self.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):

                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data

                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, b_size)
            real_data = get_slice(real_data, 1, t_size)
            fake_data = get_slice(fake_data, 0, b_size)
            fake_data = get_slice(fake_data, 1, t_size)

        else:
            real_data = real_data[:b_size, :t_size]
            fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(real_data)

        disc_interpolates = model.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty
    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1/(D-1) * X @ X.transpose(-1, -2)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        d_step = model.discrim_step(model.update_num)
        g_step = model.gen_step(model.update_num)
        
        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0
        
        if d_step:
            entropy_pen = None
            speech_encoder_out, speech_discriminator_out, speech_padding_mask = model(**sample["net_input"], input_type='speech', output_type='gan')
            text_encoder_out, text_discriminator_out, text_padding_mask = model(**sample["net_input"]["random_label"]["net_input"], input_type='text', output_type='gan')
            loss_dense = F.binary_cross_entropy_with_logits(
                speech_discriminator_out,
                speech_discriminator_out.new_ones(speech_discriminator_out.shape) - fake_smooth,
                reduction="sum",
            )
            loss_token = F.binary_cross_entropy_with_logits(
                text_discriminator_out,
                text_discriminator_out.new_zeros(text_discriminator_out.shape) + real_smooth,
                reduction="sum",
            )
            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(model, text_encoder_out, speech_encoder_out)
                grad_pen = grad_pen.sum() * self.gradient_penalty
            else:
                grad_pen = None
            sample_size =  text_encoder_out.size(0)
            
            loss = self.gan_loss_weight * (loss_dense + loss_token)  
            if grad_pen is not None:
                loss += self.gan_loss_weight * grad_pen
                
                logging_output = {
                    "discriminator_loss": loss.item(),
                    "discriminator_loss_dense": self.gan_loss_weight * loss_dense.item(),
                    "discriminator_loss_token": self.gan_loss_weight * loss_token.item(),
                    "grad_pen": self.gan_loss_weight * grad_pen.item(),
                    "discriminator_sample_size": sample_size
                }
            else:
                logging_output = {
                    "discriminator_loss": loss.item(),
                    "discriminator_loss_dense": self.gan_loss_weight * loss_dense.item(),
                    "discriminator_loss_token": self.gan_loss_weight * loss_token.item(),
                    "grad_pen": 0,
                    "discriminator_sample_size": sample_size
                }
        elif g_step:
            grad_pen = None
            speech_encoder_out, speech_discriminator_out, speech_padding_mask = model(**sample["net_input"], input_type='speech', output_type='gan')
            # print(sample["net_input"]["random_label"]["net_input"]["task_name"].keys())
            text_encoder_out, text_discriminator_out, text_padding_mask = model(**sample["net_input"]["random_label"]["net_input"], input_type='text', output_type='gan')
            speech_feats = speech_encoder_out[torch.logical_not(speech_padding_mask)]
            text_feats = text_encoder_out[torch.logical_not(text_padding_mask)]
            # print(speech_feats.size(), speech_encoder_out.size(), torch.logical_not(speech_padding_mask).sum(), 'speech')
            # print(text_feats.size(), text_encoder_out.size(), torch.logical_not(text_padding_mask).sum(), 'text')
            
            speech_variance = torch.sum(torch.log(torch.diagonal(self.cov(speech_feats))))
            text_variance = torch.sum(torch.log(torch.diagonal(self.cov(text_feats))))
            
            entropy_pen = (- speech_variance - text_variance) * self.entropy_penalty
            entropy_pen = torch.clamp(entropy_pen, min = -self.entropy_threshold).to(speech_variance.device)
            loss_dense = F.binary_cross_entropy_with_logits(
                speech_discriminator_out,
                speech_discriminator_out.new_zeros(speech_discriminator_out.shape) + fake_smooth,
                reduction="sum",
            )
            loss_token = F.binary_cross_entropy_with_logits(
                text_discriminator_out,
                text_discriminator_out.new_ones(text_discriminator_out.shape) - real_smooth,
                reduction="sum",
            )
            
            sample_size =  text_encoder_out.size(0)
            
            loss = self.gan_loss_weight * (loss_dense + loss_token  + entropy_pen)
            logging_output = {
                "generator_loss": loss.item(),
                "generator_loss_dense": self.gan_loss_weight * loss_dense.item(),
                "generator_loss_token": self.gan_loss_weight * loss_token.item(),
                "generator_loss_entropy": self.gan_loss_weight * entropy_pen.item(),
                "generator_sample_size": sample_size
            }

        else:
            assert False
            
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        generator_loss_sum = sum(log.get("generator_loss", 0) for log in logging_outputs)
        generator_dense_loss_sum = sum(log.get("generator_loss_dense", 0) for log in logging_outputs)
        generator_token_loss_sum = sum(log.get("generator_loss_token", 0) for log in logging_outputs)
        generator_entropy_loss_sum = sum(log.get("generator_loss_entropy", 0) for log in logging_outputs)
        generator_sample_size = sum(log.get("generator_sample_size", 0) for log in logging_outputs)
        discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in logging_outputs)
        discriminator_dense_loss_sum = sum(log.get("discriminator_loss_dense", 0) for log in logging_outputs)
        discriminator_token_loss_sum = sum(log.get("discriminator_loss_token", 0) for log in logging_outputs)
        discriminator_sample_size = sum(log.get("discriminator_sample_size", 0) for log in logging_outputs)
        gp_loss_sum = sum(log.get("grad_pen", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
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
            "discriminator_loss", discriminator_loss_sum /discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
        )
        metrics.log_scalar(
            "discriminator_loss_dense",discriminator_dense_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
        )
        metrics.log_scalar(
            "discriminator_loss_token", discriminator_token_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
        )
        metrics.log_scalar(
            "grad_pen", gp_loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

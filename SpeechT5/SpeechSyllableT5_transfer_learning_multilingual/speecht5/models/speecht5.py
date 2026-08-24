# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import math
from ast import literal_eval
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from .modules.text_encoder_prenet import TextEncoderPrenet
from .modules.text_decoder_prenet import TextDecoderPrenet
from .modules.text_decoder_postnet import TextDecoderPostnet
from .modules.speech_encoder_prenet import SpeechEncoderPrenet
from .modules.speech_encoder_postnet import SpeechEncoderPostnet
from .modules.speech_decoder_prenet import SpeechDecoderPrenet
from .modules.speech_decoder_postnet import SpeechDecoderPostnet
from .modules.speaker_decoder_postnet import SpeakerDecoderPostnet
from .modules.encoder import TransformerEncoder
from .modules.decoder import TransformerDecoder

# discriminators
from .modules.discriminator_wav2vecu import Discriminator as Wav2vecu_Discriminator

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    GumbelVectorQuantizer,
)
from torch import Tensor
from fairseq.data.encoders.utils import get_whole_word_mask
import copy

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_POSITIONS = 450
DEFAULT_MAX_SPEECH_POSITIONS = 4000

def print_grad(name, grad):
    print(name, grad[0].size(), grad)
    
    
class Predictor(torch.nn.Module):
    def __init__(self, args, out_dim):
        super().__init__()
        new_args = copy.deepcopy(args)
        self.predictor_net = torch.nn.Linear(args.encoder_embed_dim, out_dim)

    def forward(self, src, padding_mask):
        predictor_encoder_states = src
        x = self.predictor_net(predictor_encoder_states)
        return x



@register_model("t5_transformer")
class T5TransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(
            self, 
            args,
            encoder, decoder,
            text_encoder_prenet, speech_encoder_prenet,
            text_decoder_prenet, speech_decoder_prenet,
            text_decoder_postnet, speech_decoder_postnet,
            speaker_decoder_postnet, speech_encoder_postnet,
            text_encoder_postnet,
            discriminator, speech_mask_idx, text_mask_idx,
            speech_predictor, text_predictor,
            speech_num_tokens, text_num_tokens
        ):
        super().__init__(encoder, decoder)
        self.tot = 0
        self.encoder = encoder
        self.decoder = decoder

        self.text_encoder_prenet = text_encoder_prenet
        self.speech_encoder_prenet = speech_encoder_prenet

        self.text_decoder_prenet = text_decoder_prenet
        self.speech_decoder_prenet = speech_decoder_prenet

        self.text_decoder_postnet = text_decoder_postnet
        self.speech_decoder_postnet = speech_decoder_postnet
        self.speaker_decoder_postnet = speaker_decoder_postnet

        self.speech_encoder_postnet = speech_encoder_postnet
        self.text_encoder_postnet = text_encoder_postnet
        
        self.speech_predictor = speech_predictor
        self.text_predictor = text_predictor

        for p in self.encoder.parameters():
            p.param_group = "generator"
        for p in self.decoder.parameters():
            p.param_group = "generator"
        for p in self.text_encoder_prenet.parameters():
            p.param_group = "generator"
        for p in self.speech_encoder_prenet.parameters():
            p.param_group = "generator"
        for p in self.text_decoder_prenet.parameters():
            p.param_group = "generator"
        for p in self.speech_decoder_prenet.parameters():
            p.param_group = "generator"
        for p in self.speech_decoder_postnet.parameters():
            p.param_group = "generator" 
        for p in self.text_decoder_postnet.parameters():
            p.param_group = "generator"
        for p in self.speech_encoder_postnet.parameters():
            p.param_group = "generator"
        for p in self.text_encoder_postnet.parameters():
            p.param_group = "generator"

        self.use_codebook = args.use_codebook
        self.codebook_prob = getattr(args, "codebook_prob", 0.5) # args.codebook_prob
        if self.use_codebook:
            vq_dim = args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=args.encoder_embed_dim,
                num_vars=args.latent_vars,
                temp=args.latent_temp,
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=args.quantizer_depth,
                weight_proj_factor=args.quantizer_factor,
            )
            for p in self.quantizer.parameters():
                p.param_group = "generator" 

        self.mask_ratio = self.speech_encoder_prenet.mask_ratio
        self.interleave_ratio = 0.5
        self.interleave_span_distribution = self.speech_encoder_prenet.mask_span_distribution
        self.add_whole_word_mask = self.speech_encoder_prenet.add_whole_word_mask

        self.update_num = 0
        self.start_gan_updates = args.start_gan_updates
        
        self.discriminator = discriminator
        for p in self.discriminator.parameters():
            p.param_group = "discriminator"
        # # Follow BERT's random weight initialization (for BART)
        if args.bert_init:
            self.apply(init_bert_params)
        self.args = args
        self.prune_modules(args.modules_filter)
        self.speech_mask_idx = speech_mask_idx
        self.text_mask_idx = text_mask_idx
        self.speech_num_tokens = speech_num_tokens
        self.text_num_tokens = text_num_tokens
        
        self.baseline = torch.nn.Parameter(torch.zeros(1))
        self.baseline.requires_grad = False
        self.baseline_ema_coeff = args.baseline_ema_coeff
        
        self.turn_off_finetune_speech_enc_dec = args.turn_off_finetune_speech_enc_dec
        self.turn_off_finetune_all_enc_dec = args.turn_off_finetune_all_enc_dec

        if args.init_model_dir is not None:
            pretrained_params = torch.load(args.init_model_dir)['model']
            
            encoder_dict = {}
            
            encoder_keys = [k for k in list(pretrained_params.keys()) if 'encoder.' in k]
            
            for k in encoder_keys:
                encoder_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            speech_predictor_dict = {}
            
            speech_predictor_keys = [k for k in list(pretrained_params.keys()) if 'speech_predictor.' in k]
            
            for k in speech_predictor_keys:
                speech_predictor_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            text_predictor_dict = {}
            
            text_predictor_keys = [k for k in list(pretrained_params.keys()) if 'text_predictor.' in k]
            
            for k in text_predictor_keys:
                text_predictor_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
            decoder_dict = {}
            
            decoder_keys = [k for k in list(pretrained_params.keys()) if 'decoder.' in k]
            
            for k in decoder_keys:
                decoder_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
            speech_encoder_prenet_dict = {}

            if not args.no_load_policy:
                speech_encoder_prenet_keys = [k for k in list(pretrained_params.keys()) if ('speech_encoder_prenet.' in k and 'quantizer.embedding' not in k)]
            else:
                speech_encoder_prenet_keys = [k for k in list(pretrained_params.keys()) if (('speech_encoder_prenet.' in k) and ('quantizer.embedding' not in k) and ('policy_network' not in k) and ('frame_target_network' not in k) and ('policy_pre_proj' not in k) and ('frame_pos_enc' not in k))]
            
            for k in speech_encoder_prenet_keys:
                speech_encoder_prenet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            text_encoder_prenet_dict = {}
            
            text_encoder_prenet_keys = [k for k in list(pretrained_params.keys()) if 'text_encoder_prenet.' in k]
            
            for k in text_encoder_prenet_keys:
                text_encoder_prenet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                 
                
            speech_decoder_prenet_dict = {}
            
            speech_decoder_prenet_keys = [k for k in list(pretrained_params.keys()) if 'speech_decoder_prenet.' in k]
            
            for k in speech_decoder_prenet_keys:
                speech_decoder_prenet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            text_decoder_prenet_dict = {}
            
            text_decoder_prenet_keys = [k for k in list(pretrained_params.keys()) if 'text_decoder_prenet.' in k]
            
            for k in text_decoder_prenet_keys:
                text_decoder_prenet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            speech_decoder_postnet_dict = {}
            
            speech_decoder_postnet_keys = [k for k in list(pretrained_params.keys()) if 'speech_decoder_postnet.' in k]
            
            for k in speech_decoder_postnet_keys:
                speech_decoder_postnet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
                
            text_decoder_postnet_dict = {}
            
            text_decoder_postnet_keys = [k for k in list(pretrained_params.keys()) if 'text_decoder_postnet.' in k]
            
            for k in text_decoder_postnet_keys:
                text_decoder_postnet_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
            
            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.speech_predictor.load_state_dict(speech_predictor_dict, strict = True)
            self.text_predictor.load_state_dict(text_predictor_dict, strict = True)
            del encoder_dict
            del decoder_dict
            del speech_predictor_dict
            del text_predictor_dict
            
            
            logger.info(f'check speech prenet embed size: {self.speech_encoder_prenet.encoder_prenet[0].weight.data.size()}')
            for k in self.speech_encoder_prenet.state_dict().keys():
                if k not in speech_encoder_prenet_dict.keys():
                    logger.info(f'speech encoder prenet missing: {k}')
                    
            if not args.no_load_speech_prepost:
                self.speech_encoder_prenet.load_state_dict(speech_encoder_prenet_dict, strict=False)
            
            if not args.no_load_text_prepost:
                self.text_encoder_prenet.load_state_dict(text_encoder_prenet_dict, strict=True)
            
            if not args.no_load_speech_prepost:
                self.speech_decoder_prenet.load_state_dict(speech_decoder_prenet_dict, strict=True)
                
            if not args.no_load_text_prepost:
                self.text_decoder_prenet.load_state_dict(text_decoder_prenet_dict, strict=True)
            
            if not args.no_load_speech_prepost:
                self.speech_decoder_postnet.load_state_dict(speech_decoder_postnet_dict, strict=True)
                
            if not args.no_load_text_prepost:
                self.text_decoder_postnet.load_state_dict(text_decoder_postnet_dict, strict=True)  
                
            del speech_encoder_prenet_dict
            del text_encoder_prenet_dict
            del speech_decoder_prenet_dict
            del text_decoder_prenet_dict
            del speech_decoder_postnet_dict
            del text_decoder_postnet_dict
            
            
            quantizer_dict = {}
            
            quantizer_keys = [k for k in list(pretrained_params.keys()) if 'quantizer.' in k and 'prenet' not in k]
            
            for k in quantizer_keys:
                quantizer_dict[k[k.find('.')+1:]] = pretrained_params.pop(k)
                
            if not args.no_load_quantizer:
                self.quantizer.load_state_dict(quantizer_dict, strict=True)
                
            del quantizer_dict
                
            
            logger.info(f'Loaded weights from {args.init_model_dir}')


        # for name, layer in self.speech_encoder_prenet.named_modules():
        #     # print(name)
        #     layer.register_backward_hook(lambda module, grad_input, grad_output: print_grad(name, grad_output))
        
        if self.turn_off_finetune_all_enc_dec:
            for i, param in enumerate(self.encoder.parameters()):
                param.requires_grad = False
            
            for i, param in enumerate(self.decoder.parameters()):
                param.requires_grad = False
            
            for i, param in enumerate(self.text_encoder_prenet.parameters()):
                param.requires_grad = False
            
            for i, param in enumerate(self.text_decoder_prenet.parameters()):
                param.requires_grad = False
                
            for i, param in enumerate(self.text_decoder_postnet.parameters()):
                param.requires_grad = False
    
            for i, param in enumerate(self.text_predictor.parameters()):
                param.requires_grad = False

        
        # torch.autograd.set_detect_anomaly(True)
        
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--reduction-factor",
            type=int,
            help="reduction factor for decoder",
        )
        parser.add_argument(
            "--spk-embed-dim",
            type=int,
            help="speaker embedding dimension",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            '--freeze-encoder-updates',
            type=int,
            help='number of steps to freeze encoder before finetune'
        )
        parser.add_argument(
            '--freeze-decoder-updates',
            type=int,
            help='number of steps to freeze decoder before finetune'
        )
        parser.add_argument(
            '--no-freeze-encoder-layer',
            type=str,
            help='which encoder layer not freeze during finetune'
        )
        parser.add_argument(
            "--share-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-ctc-embed",
            action="store_true",
            help="share ctc embed and decoder embed",
        )
        parser.add_argument(
            "--encoder-sliding-window-attn",
            default=None,
            type=int,
            help="If not None but a even number, set sliding window attention to encoder's attn_mask, e.g., 4, 10, and 20",
        )
        
        # Convolutional subsampler
        parser.add_argument(
            "--encoder-speech-prenet",
            default="linear",
            type=str,
            choices=["conv", "linear"],
            help="The type of encoder speech prenet, e.g., conv or linear."
        )
        parser.add_argument(
            "--conv-kernel-sizes",
            default="5,5",
            type=str,
            help="The layer of convolution of encoder speech prenet."
        )
        parser.add_argument(
            "--conv-channels",
            default=1024,
            type=int,
            help="The channels of encoder speech prenet."
        )
        parser.add_argument(
            "--input-dim",
            default=768,
            type=int,
            help="The dimension of input"
        )
        parser.add_argument(
            "--encoder-prenet-embed-dim",
            default=256,
            type=int,
            help="The dimension of embedding"
        )
        parser.add_argument(
            "--subsample-stride",
            default="2,2",
            type=str,
            help="The subsample stride for conv1dsubsample."
        )
        parser.add_argument(
            "--spk-embed-integration-type",
            type=str,
            choices=["pre", "add"],
            help="speaker embedding integration type"
        )
        parser.add_argument(
            "--dprenet-dropout-rate",
            default=0.5,
            type=float,
            help="The dropout rate of decoder speech prenet."
        )
        
        ## SE
        parser.add_argument(
            "--se-predict",
            default=None,
            choices=["masking", "target", "delta"],
            help="If set, source speech inputs decoder to predict the masking/target/delta of corresponding inputs."
               + "masking is [0, 1], target is predicted output, delta is difference between inputs and outputs",
        )
        parser.add_argument(
            "--se-decoder-input",
            type=str,
            default="previous_target",
            choices=["previous_target", "source"],
        )
        
        ## SID
        parser.add_argument(
            "--modules-filter",
            default=None,
            type=str,
            help="Remove unused modules for, e.g., SID.",
        )
        parser.add_argument(
            "--sid-pad-prenet",
            action="store_true",
            help="If set, the size of text dictionary is as small as for <pad> token.",
        )
        parser.add_argument(
            "--encoder-attn-branch",
            type=str,
            default="identity,full",
            help="encoder attention branch sliding window, e.g., 'identity,0,2,4,full'",
        )
        parser.add_argument(
            "--encoder-block-branch",
            type=str,
            help="average the output of encoder, e.g., '4,5,6'",
        )
        parser.add_argument(
            "--sid-encoder-cls",
            default=None,
            choices=["encoder"],
            help="If set, add cls vector to the encoder input, e.g., constant vector.",
        )
        parser.add_argument(
            "--sid-shuffle-encoder-input",
            action="store_true",
            help="If set, shuffle encoder input in time.",
        )
        parser.add_argument(
            "--sid-decoder-speaker",
            action="store_true",
            help="If set, apply speaker decoder as transformer decoder.",
        )
        parser.add_argument(
            "--sid-decoder-attn-dim",
            default=128,
            type=int,
            help="Attention dimension in attensive statistics pooling of speaker decoder.",
        )
        parser.add_argument(
            "--sid-t5-postnet",
            action="store_true",
            help="If set, apply TextDecoderPostnet as speaker classification.",
        )
        parser.add_argument(
            "--sid-embed-dim",
            default=128,
            type=int,
            help="Embedding dimension in speaker postnet for speaker identification if embed postnet.",
        )
        parser.add_argument(
            "--sid-pooling-layer",
            default="decoder",
            type=str,
            choices=["decoder-las", "decoder", "encoder", "encoder-cls", "encoder-speaker"],
            help="The output of decoder or encoder uses as SID pooling layer over temporal dimension.",
        )
        parser.add_argument(
            "--sid-no-pooling-bn",
            action="store_true",
            help="If set, not attention batchnorm.",
        )
        parser.add_argument(
            "--sid-no-embed-postnet",
            action="store_true",
            help="If set, no layer between decoder output and classification layer.",
        )
        parser.add_argument(
            "--sid-normalize-postnet",
            action="store_true",
            help="If set, normalize input and weight in postnet/classifier.",
        )
        parser.add_argument(
            "--sid-softmax-type",
            default="softmax",
            choices=["softmax", "amsoftmax", "aamsoftmax"],
            help="If using amsoftmax or aamsoftmax, the target should be given.",
        )
        parser.add_argument(
            "--softmax-scale",
            default=1.0,
            type=float,
            help="Scale for AMSoftmax or AAMSoftmax.",
        )
        parser.add_argument(
            "--softmax-margin",
            default=0.0,
            type=float,
            help="Margin for AMSoftmax or AAMSoftmax.",
        )
        parser.add_argument(
            "--softmax-easy-margin",
            action="store_true",
            help="Enable easy margin for AAMSoftmax.",
        )
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--decoder-layerdrop",
            type=float,
            metavar="D",
            help="LayerDrop probability for decoder",
        )
        
        ## Hubert
        parser.add_argument(
            '--feature-grad-mult',
            type=float,
            help='multiply feature extractor var grads by this'
        )
        parser.add_argument(
            '--logit-temp',
            type=float,
            help='temperature to divide logits by'
        )
        parser.add_argument(
            '--final-dim',
            type=int,
            help="project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        )
        
        # mask
        parser.add_argument(
            '--hubert-mask-length',
            type=int,
            help='mask length'
        )
        parser.add_argument(
            '--mask-prob',
            type=float,
            help='probability of replacing a token with mask'
        )
        parser.add_argument(
            "--mask-selection",
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose mask length",
        )
        parser.add_argument(
            '--mask-other',
            type=float,
            help="secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        )
        parser.add_argument(
            '--mask-min-space',
            type=int,
            help='min space between spans (if no overlap is enabled)'
        )
        
        # channel masking
        parser.add_argument(
            '--mask-channel-length',
            type=int,
            help='length of the mask for features (channels)'
        )
        parser.add_argument(
            '--mask-channel-prob',
            type=float,
            help="probability of replacing a feature with 0"
        )
        parser.add_argument(
            "--mask-channel-selection",
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose mask length for channel masking",
        )
        parser.add_argument(
            '--mask-channel-other',
            type=float,
            help="secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        )
        parser.add_argument(
            '--mask-channel-min-space',
            type=int,
            help='min space between spans (if no overlap is enabled)'
        )
        
        # abs positional embeddings
        parser.add_argument(
            '--conv-pos',
            type=int,
            help='number of filters for convolutional positional embeddings'
        )
        parser.add_argument(
            '--conv-pos-groups',
            type=int,
            help='number of groups for convolutional positional embedding'
        )
        
        # codebook related
        parser.add_argument(
            "--use-codebook",
            action="store_true",
            help="whether to use codebook",
        )
        parser.add_argument(
            "--codebook-prob",
            type=float,
            help="probability to use codebook",
        )
        parser.add_argument(
            "--latent-vars",
            type=int,
            help="number of latent variables V in each group of the codebook",
        )
        parser.add_argument(
            "--latent-groups",
            type=int,
            help="number of groups G of latent variables in the codebook",
        )
        parser.add_argument(
            "--latent-dim",
            type=int,
            help="if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups",
        )
        parser.add_argument(
            "--latent-temp",
            type=literal_eval,
            help="temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)",
        )
        parser.add_argument(
            "--quantizer-depth",
            type=int,
            help="number of quantizer layers",
        )
        parser.add_argument(
            "--quantizer-factor",
            type=int,
            help="number of quantizer layers",
        )
        parser.add_argument(
            "--get-code-distribution",
            action='store_true',
            help="whether to get the code distribution (for test)",
        )

        # relative pos enc
        parser.add_argument(
            "--relative-position-embedding",
            action='store_true',
            help="whether to use relative position embedding",
        )
        parser.add_argument(
            "--num-buckets",
            type=int,
            default=320,
            help="num of buckets for relative position embedding",
        )
        parser.add_argument(
            "--max-distance",
            type=int,
            default=1280,
            help="max distance for relative position embedding",
        )
        parser.add_argument(
            "--encoder-max-relative-position",
            type=int,
            help="max distance for relative position embedding in encoder",
        )
        parser.add_argument(
            "--decoder-max-relative-position",
            type=int,
            help="max distance for relative position embedding in decoder",
        )

        # hubert feature extractor
        parser.add_argument(
            "--conv-feature-layers",
            type=str,
            help= "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]",
        )
        parser.add_argument(
            "--conv-bias",
            action='store_true',
            help="include bias in conv encoder",
        )
        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        )

        # others
        parser.add_argument(
            "--bert-init",
            action='store_true',
            help="initilize as bert",
        )
        parser.add_argument(
            "--unb-enc-layer",
            type=int,
            default=-1,
            help="which layer's output is used as the input of decoder",
        )
        
        # discriminator related
        parser.add_argument(
            "--discriminator-arch",
            default="wav2vecu",
            type=str,
            choices=["wav2vecu", "largecnn","transformer"],
            help="The type of discriminator to use"
        )
        parser.add_argument(
            "--start-gan-updates",
            default=5000,
            type=int,
            help="Start GAN update after this many steps"
        )
        
        #   discriminator_dim: 384
        #   discriminator_depth: 2
        #   discriminator_kernel: 8
        #   discriminator_linear_emb: false
        #   discriminator_causal: true /////////////
        #   discriminator_max_pool: false
        #   discriminator_act_after_linear: false
        #   discriminator_dropout: 0.0
        #   discriminator_weight_norm: false
        # wav2vecu discriminator related parameters
        parser.add_argument(
            "--discriminator-dim",
            default=768,
            type=int,
            help="discriminator CNN dimension"
        )
        
        parser.add_argument(
            "--discriminator-kernel",
            default=8,
            type=int,
            help="discriminator CNN kernel size"
        )
        
        parser.add_argument(
            "--discriminator-dilation",
            default=1,
            type=int,
            help="discriminator CNN dilation size"
        )
        
        parser.add_argument(
            "--discriminator-max-pool",
            action='store_true',
            help="whether to use max pooling or mean pooling across time"
        )
        
        parser.add_argument(
            "--discriminator-causal",
            action='store_true',
            help="whether to use causal convolution for discriminator"
        )
        
        parser.add_argument(
            "--discriminator-spectral-norm",
            action='store_true',
            help="whether to use spectral normalization for discriminator"
        )
        
        parser.add_argument(
            "--discriminator-weight-norm",
            action='store_true',
            help="whether to use weight norm for discriminator"
        )
        
        parser.add_argument(
            "--discriminator-dropout",
            default=0.0,
            type=float,
            help="discriminator CNN dropout probability"
        )
        
        parser.add_argument(
            "--discriminator-depth",
            default=3,
            type=int,
            help="number of CNN layers in the discriminator"
        )        
        
        parser.add_argument(
            "--discriminator-linear-emb",
            action='store_true',
            help="whether to use a linear layer or a CNN layer as first layer of discriminator"
        )
        
        parser.add_argument(
            "--discriminator-act-after-linear",
            action='store_true',
            help="whether to use GELU after the first layer of discriminator"
        )
        
        parser.add_argument(
            "--use-softmax-soft-pool",
            action='store_true',
            help="whether to use GELU after the first layer of discriminator"
        )
        parser.add_argument(
            "--speech-encoder-prenet-mask",
            default=0.3,
            type=float,
            help="fraction of speech words/subwords that will be masked",
        )
        parser.add_argument(
            "--speech-encoder-prenet-mask-random",
            default=0.1,
            type=float,
            help="instead of using [MASK], use random token this often for speech",
        )
        parser.add_argument(
            "--speech-encoder-prenet-mask-length",
            default="span-poisson",
            type=str,
            choices=["subword", "word", "span-poisson"],
            help="mask length to choose for speech",
        )
        parser.add_argument(
            "--speech-encoder-prenet-poisson-lambda",
            default=3.5,
            type=float,
            help="randomly shuffle sentences for this proportion of inputs for speech",
        )
        parser.add_argument(
            "--speech-encoder-prenet-replace-length",
            default=1,
            type=int,
            help="when masking N speech tokens, replace with 0, 1, or N tokens (use -1 for N)",
        )
        parser.add_argument(
            "--speech-encoder-prenet-iid-noise-target",
            action="store_true",
            help="whether to use t5 form target for speech",
        )
        parser.add_argument(
            "--word-freq",
            default=12,
            type=float,
            help="how many frames per word",
        )
        parser.add_argument(
            "--km-size",
            type=int,
            default=1025,
            help="number of speech clusters"
        )
        parser.add_argument(
            "--num-lang",
            type=int,
            default=3,
            help="number of languages"
        )
        parser.add_argument(
            "--frame-target-classes",
            type=int,
            default=100,
            help="number of frame-level target clusters"
        )
        parser.add_argument(
            "--init-clus-dir",
            type=str,
            help="from where to initialize speech cluster centroids"
        )
        parser.add_argument(
            "--init-encoder-path",
            type=str,
            help="from where to initialize the encoder"
        )
        parser.add_argument(
            "--init-model-dir",
            type=str,
            help="from where to initialize the encoder, decoder and prenets"
        )
        parser.add_argument(
            "--sampling-nums",
            type=int,
            default=3,
            help="number of sampling operations to perform"
        )       
        parser.add_argument(
            "--baseline-ema-coeff",
            type=float,
            default=0.99,
            help="ema coefficient for baseline reward"
        )
        parser.add_argument(
            "--turn-off-finetune-speech-enc-dec",
            action='store_true',
            help="whether to turn off gradient updates to enc-dec during speech finetuning"
        )
        parser.add_argument(
            "--turn-off-finetune-all-enc-dec",
            action='store_true',
            help="whether to turn off gradient updates to enc-dec during speech/text joint finetuning"
        )
        parser.add_argument(
            "--use-transformer-policy",
            action='store_true',
            help="whether to use a transformer layer for segmentation policy"
        )
        parser.add_argument(
            "--no-load-policy",
            action='store_true',
            help="whether to load pretrained policy"
        )
        parser.add_argument(
            "--no-load-speech-prepost",
            action='store_true',
            help="whether to load speech prenets and postnets"
        )
        parser.add_argument(
            "--no-load-text-prepost",
            action='store_true',
            help="whether to load text prenets and postnets"
        )
        parser.add_argument(
            "--no-load-quantizer",
            action='store_true',
            help="whether to load text prenets and postnets"
        )
        # no_load_speech_prepost:
        parser.add_argument(
            "--policy-feat-dim",
            type=int,
            default=768,
            help="number of sampling operations to perform"
        )    

    # Encoder, Decoder
    @classmethod
    def build_encoder(cls, args, dictionary=None, embed_tokens=None, src_dictionary=None, src_embed_tokens=None):
        return TransformerEncoder(args, dictionary, embed_tokens, src_dictionary, src_embed_tokens)

    @classmethod
    def build_decoder(cls, args):
        return TransformerDecoder(args)

    # Encoder Prenet
    @classmethod
    def build_text_encoder_prenet(cls, embed_tokens, args):
        return TextEncoderPrenet(embed_tokens, args)

    # Encoder Prenet
    @classmethod
    def build_speech_encoder_prenet(cls, args, embed_tokens, vocab, mask_whole_word, mask_idx, frame_target_classes):
        return SpeechEncoderPrenet(args, embed_tokens, vocab, mask_whole_word, mask_idx, frame_target_classes, sampling_nums=args.sampling_nums,
                                   policy_feat_dim=args.policy_feat_dim,
                                   num_lang=args.num_lang,
                                   use_transformer_policy=args.use_transformer_policy,
                                   use_softmax_soft_pool=args.use_softmax_soft_pool,
                                   mask_length = args.speech_encoder_prenet_mask_length, poisson_lambda = args.speech_encoder_prenet_poisson_lambda,
                                   mask = args.speech_encoder_prenet_mask, mask_random =args.speech_encoder_prenet_mask_random, iid_noise_target = args.speech_encoder_prenet_iid_noise_target, replace_length=args.speech_encoder_prenet_replace_length,word_freq=args.word_freq)
    # Decoder Prenet
    @classmethod
    def build_text_decoder_prenet(cls, embed_tokens, args):
        return TextDecoderPrenet(embed_tokens, args)

    # Decoder Prenet
    @classmethod
    def build_speech_decoder_prenet(cls, embed_tokens, args):
        return TextDecoderPrenet(embed_tokens, args)

    # Decoder Postnet
    @classmethod
    def build_text_decoder_postnet(cls, embed_tokens, dictionary, args):
        return TextDecoderPostnet(embed_tokens, dictionary, args)

    @classmethod
    def build_speaker_decoder_postnet(cls, embed_dim, class_num, args):
        return SpeakerDecoderPostnet(embed_dim, class_num, args)

    # Decoder Postnet
    @classmethod
    def build_speech_decoder_postnet(cls, embed_tokens, dictionary, args):
        return TextDecoderPostnet(embed_tokens, dictionary, args)

    @classmethod
    def build_speech_encoder_postnet(cls, dictionaries, args):
        return SpeechEncoderPostnet(dictionaries, args)
    
    @classmethod
    def build_discriminator(cls, args):
        if args.discriminator_arch == 'wav2vecu':
            return Wav2vecu_Discriminator(args)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim, max_num_embeddings=None):
            num_embeddings = len(dictionary)
            if max_num_embeddings is not None and isinstance(max_num_embeddings, int):
                num_embeddings = min(num_embeddings, max_num_embeddings)  
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if hasattr(args, "sid_pad_prenet") and args.sid_pad_prenet:
            max_num_embeddings = 3 # <pad> at index 2
        else:
            max_num_embeddings = None

        # print('start building model')
        text_decoder_embed_tokens = build_embedding(
            task.dicts["text"], args.decoder_embed_dim, max_num_embeddings
        )        
        
        speech_decoder_embed_tokens = build_embedding(
            task.dicts["audio"], args.decoder_embed_dim, max_num_embeddings
        )        
        
        if args.share_input_output_embed:
            text_encoder_embed_tokens = text_decoder_embed_tokens
        else:
            text_encoder_embed_tokens = build_embedding(
                task.dicts["text"], args.encoder_embed_dim
            )
            
        if args.share_input_output_embed:
            speech_encoder_embed_tokens = speech_decoder_embed_tokens
        else:
            speech_encoder_embed_tokens = build_embedding(
                task.dicts["audio"], args.encoder_embed_dim
            )
            
        if "audio" in task.dicts and "text" in task.dicts:
            encoder =  cls.build_encoder(args, task.dicts["text"], text_encoder_embed_tokens, task.dicts["audio"], speech_encoder_embed_tokens)
        elif "text" in task.dicts:
            encoder = cls.build_encoder(args, task.dicts["text"], text_encoder_embed_tokens)
        else:
            encoder = cls.build_encoder(args)      
        decoder = cls.build_decoder(args)


        mask_whole_words = (
                get_whole_word_mask(task.args, task.dicts["audio"])
                if task.args.mask_length != "subword"
                else None
            )
        text_encoder_prenet = cls.build_text_encoder_prenet(text_encoder_embed_tokens, args)
        speech_encoder_prenet = cls.build_speech_encoder_prenet(args, speech_encoder_embed_tokens, task.dicts["audio"], mask_whole_words, task.dicts["audio"].index("<mask>"), args.frame_target_classes)

        text_decoder_prenet = cls.build_text_decoder_prenet(text_decoder_embed_tokens, args)
        speech_decoder_prenet = cls.build_text_decoder_prenet(speech_decoder_embed_tokens, args)

        text_decoder_postnet = cls.build_text_decoder_postnet(text_decoder_embed_tokens, task.dicts['text'], args)
        speech_decoder_postnet = cls.build_speech_decoder_postnet(speech_decoder_embed_tokens, task.dicts['audio'], args)

        if getattr(args, "sid_t5_postnet", False):
            speaker_decoder_postnet = None
        else:
            if task.t5_task == "s2c":
                speaker_decoder_postnet = cls.build_speaker_decoder_postnet(args.sid_embed_dim, len(task.dicts['text']), args)
            else:
                speaker_decoder_postnet = None

        speech_encoder_postnet = cls.build_speech_encoder_postnet([task.dicts['audio']],  args)
        text_encoder_postnet = cls.build_speech_encoder_postnet([task.dicts['text']], args) 

        speech_predictor = Predictor(args, len(task.dicts['audio']))
        text_predictor = Predictor(args, len(task.dicts['text']))
            
        discriminator = cls.build_discriminator(args)
        speech_num_tokens = len(task.dicts['audio'])
        text_num_tokens = len(task.dicts['text'])
        # print('finish building model')
        return cls(
            args, 
            encoder, decoder, 
            text_encoder_prenet, speech_encoder_prenet,
            text_decoder_prenet, speech_decoder_prenet,
            text_decoder_postnet, speech_decoder_postnet,
            speaker_decoder_postnet, speech_encoder_postnet,
            text_encoder_postnet,
            discriminator, task.dicts["audio"].index("<mask>"),task.dicts["text"].index("<mask>"),
            speech_predictor, text_predictor,
            speech_num_tokens, text_num_tokens
        )

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_for_ctc(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out_for_ctc"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, sample, net_output, is_masked=True):
        if "logit_m_list" in net_output:
            logits_list = self.get_logits(net_output, is_masked)
            targets_list = [
                x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
            ]
            return targets_list
        else:
            return sample["target"]

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if isinstance(net_output, list):
            features_pen_loss = []
            prob_perplexities = []
            for curr_idx, curr_net_output in enumerate(net_output):
                if "features_pen" in curr_net_output:
                    features_pen_loss.append(curr_net_output["features_pen"])
                if "prob_perplexity" in curr_net_output:
                    prob_perplexities.append((curr_net_output["num_vars"] - curr_net_output["prob_perplexity"])/ curr_net_output["num_vars"])
            if len(features_pen_loss) != 0:
                extra_losses.append(sum(features_pen_loss))
                names.append("features_pen")

            if len(prob_perplexities) != 0:
                extra_losses.append(sum(prob_perplexities) / len(prob_perplexities))
                names.append("prob_perplexity")
        else:
            if "features_pen" in net_output:
                extra_losses.append(net_output["features_pen"])
                names.append("features_pen")

            if "prob_perplexity" in net_output:
                extra_losses.append(
                    (net_output["num_vars"] - net_output["prob_perplexity"])
                    / net_output["num_vars"]
                )
                names.append("prob_perplexity")

        return extra_losses, names
    
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        
    def get_groups_for_update(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates) else "generator"

    def interleave_and_mask_speech_text(self, sources, texts, p):
        mixed_list = []
        unmasked_mixed_list = []
        to_keep_list = []
        for source, text in zip(sources, texts):
            speech_random_idx = torch.randperm(source.size(0))[:int(source.size(0) * p)]
            # Make weight for q
            to_keep = source.new_zeros(source.size(0)).bool()
            to_keep[speech_random_idx] = 1

            # replace index with text
            source[~to_keep] = text[~to_keep]

            mixed, unmasked_mixed, _ = self.add_whole_word_mask(source, self.mask_ratio)
            mixed_list.append(mixed)
            unmasked_mixed_list.append(unmasked_mixed)
            to_keep_list.append(to_keep)

        mixed = torch.stack(mixed_list)
        unmasked_mixed = torch.stack(unmasked_mixed_list)
        to_keep = torch.stack(to_keep_list)

        return mixed, unmasked_mixed, to_keep

    def forward(self, source=None, lang_id=None, aux_frame_clus=None, audio_tokens = None, discrete_batch = None, src_tokens=None, prev_output_tokens=None, random_label= None,random_src_tokens=None, boundaries = None, task_name=None, padding_mask=None, only_ctc=False, feature_only=False, tgt_enc_layer=None, input_type=None, output_type=None, speech_prenet_mode=None):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        assert source is not None or src_tokens is not None        
        
        if task_name =='s2t':
            # print(audio_tokens)
            input_type ='speech'
            output_type = 'text'
            codebook_out = {}
            
            speech_meta_info = {}
            # Encoder Prenet
            if speech_prenet_mode == 'policy_pretrain':
                batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, padding_mask=padding_mask, boundaries = boundaries, discrete_batch = discrete_batch, audio_tokens = audio_tokens, use_boundaries = True, task_name=task_name)
            else:
                batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, padding_mask=padding_mask, boundaries = boundaries,discrete_batch = None, audio_tokens = None, use_boundaries = False, task_name=task_name)

            batch = batches[0]
            speech_encoder_input = batch["prenet_out"]
            speech_encoder_padding_mask = batch["padding_mask"]
            speech_meta_info["ntokens"] = batch["ntokens"]
            speech_meta_info["nsentences"] = batch["nsentences"]
            speech_meta_info["net_input"] = {}
            if speech_prenet_mode == 'policy_pretrain':
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"]
            else:
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"].argmax(-1)
            speech_meta_info["net_input"]["src_lengths"] = batch["net_input"]["src_lengths"]
            speech_meta_info["policy_logits"] = policy_logits
            speech_meta_info["frame_target_logits"] = frame_target_logits
            speech_meta_info["policy_encoder_padding_mask"] = policy_encoder_padding_mask
            speech_meta_info["target_padding_mask"] = batch["target_padding_mask"]
            speech_src_lengths = batch["net_input"]["src_lengths"]
            
            encoder_output = self.encoder(speech_encoder_input, speech_encoder_padding_mask, tgt_layer=tgt_enc_layer)
            if task_name is not None and feature_only:
                return encoder_output["encoder_out"][0].transpose(0, 1)
            
            if "decoder_input" in encoder_output and encoder_output["decoder_input"][0] is not None:
                # Change the encoder output to decoder input once set unb-enc-layer
                encoder_output["encoder_out"] = encoder_output["decoder_input"]
                
            if only_ctc and task_name is not None and task_name == "s2t":
                return None, encoder_output
            elif not self.training and prev_output_tokens is None and task_name == "s2t" and task_name is not None:
                return encoder_output
            
            if task_name is not None and task_name == 's2t':
                return (self.text_predictor(encoder_output["encoder_out"][0].transpose(0, 1), speech_encoder_padding_mask), batch["net_input"]["src_lengths"]), encoder_output

        if task_name == "speech_pretrain" and feature_only == True:
            input_type = 'speech'
            speech_meta_info = {}
            # Encoder Prenet
            if speech_prenet_mode == 'policy_pretrain':
                batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, padding_mask=padding_mask, boundaries = boundaries, discrete_batch = discrete_batch, audio_tokens = audio_tokens, use_boundaries = True, task_name=task_name)
            else:
                batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, lang_id=lang_id, padding_mask=padding_mask, boundaries = boundaries, discrete_batch = discrete_batch, audio_tokens = audio_tokens, use_boundaries = False, task_name=task_name)  # TODO
            
            batch = batches[0]   
            speech_encoder_input = batch["prenet_out"]
            speech_encoder_padding_mask = batch["padding_mask"]
            speech_meta_info["ntokens"] = batch["ntokens"]
            speech_meta_info["target"] = batch["target"]
            speech_meta_info["no_boseos_target"] = batch["no_boseos_target"]
            speech_meta_info["unmasked_padding_mask"] = batch["unmasked_padding_mask"]
            speech_meta_info["nsentences"] = batch["nsentences"]
            speech_meta_info["net_input"] = {}
            if speech_prenet_mode == 'policy_pretrain':
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"]
            else:
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"].argmax(-1)
            speech_meta_info["net_input"]["src_lengths"] = batch["net_input"]["src_lengths"]
            speech_meta_info["policy_logits"] = policy_logits
            speech_meta_info["frame_target_logits"] = frame_target_logits
            speech_meta_info["policy_encoder_padding_mask"] = policy_encoder_padding_mask
            speech_meta_info["target_padding_mask"] = batch["target_padding_mask"]
            if "hard_policy" in batch.keys():
                speech_meta_info["hard_policy"] = batch["hard_policy"]
            if "sum_after_mean_word_probs" in batch.keys():
                speech_meta_info["sum_after_mean_word_probs"] = batch["sum_after_mean_word_probs"]
            if "sum_after_mean_nonword_probs" in batch.keys():
                speech_meta_info["sum_after_mean_nonword_probs"] = batch["sum_after_mean_nonword_probs"] 
            if "sum_after_mean_consecutive_probs" in batch.keys():    
                speech_meta_info["sum_after_mean_consecutive_probs"] = batch["sum_after_mean_consecutive_probs"]
            speech_prev_output_tokens = batch["net_input"]["prev_output_tokens"]
            speech_src_lengths = batch["net_input"]["src_lengths"]
            speech_target_list = batch["net_input"]["target_list"]
            if speech_prenet_mode == 'policy_pretrain':
                speech_meta_info["src_tokens"] = batch["net_input"]["src_tokens"]
                speech_meta_info["aux_target"] = batch["net_input"]["aux_target"]
            else:
                speech_meta_info["src_tokens"] = batch["net_input"]["src_tokens"].argmax(-1)

            encoder_output = self.encoder(speech_encoder_input, speech_encoder_padding_mask, tgt_layer=tgt_enc_layer)
            return encoder_output["encoder_out"][0].transpose(0, 1), speech_meta_info
        
        if task_name == "text_pretrain" and feature_only == True:
            input_type = 'text'
            encoder_input, encoder_padding_mask = self.text_encoder_prenet(src_tokens)
            encoder_output = self.encoder(encoder_input, encoder_padding_mask, tgt_layer=tgt_enc_layer)
            return encoder_output["encoder_out"][0].transpose(0, 1)
        
        if speech_prenet_mode == 'policy_pretrain':
            batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, padding_mask=padding_mask, boundaries = boundaries, discrete_batch = discrete_batch, audio_tokens = audio_tokens, use_boundaries = True, task_name=task_name)
        else:
            batches, policy_logits, frame_target_logits, policy_encoder_padding_mask = self.speech_encoder_prenet(src=source, padding_mask=padding_mask, boundaries = boundaries, discrete_batch = None, audio_tokens = None, use_boundaries = False, task_name=task_name)

        if self.turn_off_finetune_speech_enc_dec:
            for i, param in enumerate(self.encoder.parameters()):
                param.requires_grad = False
            
            for i, param in enumerate(self.decoder.parameters()):
                param.requires_grad = False
        
        ret_speech_decoder_outputs = []
        ret_speech_codebook_outputs = []
        ret_speech_masked_lm_results = []
        ret_loss_aux_target = []
        ret_speech_meta_info  = []
        for batch in batches:
            speech_codebook_out = {}
            speech_meta_info = {}
            
            speech_encoder_input = batch["prenet_out"]
            speech_encoder_padding_mask = batch["padding_mask"]
            speech_meta_info["ntokens"] = batch["ntokens"]
            speech_meta_info["target"] = batch["target"]
            speech_meta_info["no_boseos_target"] = batch["no_boseos_target"]
            speech_meta_info["unmasked_prenet_out"] = batch["unmasked_prenet_out"]
            speech_meta_info["unmasked_padding_mask"] = batch["unmasked_padding_mask"]
            speech_meta_info["nsentences"] = batch["nsentences"]
            speech_meta_info["net_input"] = {}
            if speech_prenet_mode == 'policy_pretrain':
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"]
            else:
                speech_meta_info["net_input"]["src_tokens"] = batch["net_input"]["src_tokens"].argmax(-1)
            speech_meta_info["net_input"]["src_lengths"] = batch["net_input"]["src_lengths"]
            speech_meta_info["policy_logits"] = policy_logits
            speech_meta_info["frame_target_logits"] = frame_target_logits
            speech_meta_info["policy_encoder_padding_mask"] = policy_encoder_padding_mask
            speech_meta_info["target_padding_mask"] = batch["target_padding_mask"]
            if "hard_policy" in batch.keys():
                speech_meta_info["hard_policy"] = batch["hard_policy"]
            if "code_loss" in batch.keys():
                speech_meta_info["code_loss"] = batch["code_loss"]
            if "sum_after_mean_word_probs" in batch.keys():
                speech_meta_info["sum_after_mean_word_probs"] = batch["sum_after_mean_word_probs"]
            if "sum_after_mean_nonword_probs" in batch.keys():
                speech_meta_info["sum_after_mean_nonword_probs"] = batch["sum_after_mean_nonword_probs"] 
            if "sum_after_mean_consecutive_probs" in batch.keys():    
                speech_meta_info["sum_after_mean_consecutive_probs"] = batch["sum_after_mean_consecutive_probs"]
            
            speech_prev_output_tokens = batch["net_input"]["prev_output_tokens"]
            speech_src_lengths = batch["net_input"]["src_lengths"]
            speech_target_list = batch["net_input"]["target_list"]
            if discrete_batch is not None:
                speech_meta_info["aux_target"] = discrete_batch["net_input"]["aux_target"]

            ret_speech_meta_info.append(speech_meta_info)

            # Encoder: T x B x C
            speech_encoder_output = self.encoder(speech_encoder_input, speech_encoder_padding_mask, tgt_layer=tgt_enc_layer)

            if self.use_codebook and self.training:
                speech_q = self.quantizer(speech_encoder_output["encoder_out"][0].transpose(0, 1))
                
                # q["x"]: B x T x C
                # Sample indexs according to the codebook prob
                speech_random_idx = torch.randperm(speech_q["x"].size(1))[:int(speech_q["x"].size(1) * self.codebook_prob)]
                # Make weight for q
                speech_q_w = speech_q["x"].new_zeros(speech_q["x"].size(1))
                speech_q_w[speech_random_idx] = 1.0
                # Combine quantized codes and encoder output
                speech_encoder_output["encoder_out"][0] = (
                    speech_q_w.view(-1, 1) * speech_q["x"] + (- speech_q_w + 1).view(-1, 1) * speech_encoder_output["encoder_out"][0].transpose(0, 1)
                ).transpose(0, 1)

                speech_codebook_out["prob_perplexity"] = speech_q["prob_perplexity"]
                speech_codebook_out["code_perplexity"] = speech_q["code_perplexity"]
                speech_codebook_out["num_vars"] = speech_q["num_vars"]
                speech_codebook_out["temp"] = speech_q["temp"]

            ret_speech_masked_lm_results.append((self.speech_predictor(speech_encoder_output["encoder_out"][0].transpose(0, 1), speech_encoder_padding_mask), None))
            ret_speech_codebook_outputs.append(speech_codebook_out)

        masked_text_lm_results = None 
        ret_unigram_loss_speech = None
        ret_unigram_loss_text = None
        
        if (self.turn_off_finetune_speech_enc_dec) and (not self.turn_off_finetune_all_enc_dec):
            for i, param in enumerate(self.encoder.parameters()):
                param.requires_grad = True
            
            for i, param in enumerate(self.decoder.parameters()):
                param.requires_grad = True
        
        prev_output_tokens = random_label["net_input"]["prev_output_tokens"]
        target_list = random_label["net_input"]["target_list"]
        src_tokens = random_label["net_input"]["src_tokens"]
        
        text_codebook_out = {}
        
        # Encoder Prenet
        text_encoder_input, text_encoder_padding_mask = self.text_encoder_prenet(src_tokens)
        text_encoder_output = self.encoder(text_encoder_input, text_encoder_padding_mask, tgt_layer=tgt_enc_layer)
        if "decoder_input" in text_encoder_output and text_encoder_output["decoder_input"][0] is not None:
            # Change the encoder output to decoder input once set unb-enc-layer
            text_encoder_output["encoder_out"] = text_encoder_output["decoder_input"]
            
        if self.use_codebook and self.training:
            text_q = self.quantizer(text_encoder_output["encoder_out"][0].transpose(0, 1))
            
            # q["x"]: B x T x C
            # Sample indexs according to the codebook prob
            text_random_idx = torch.randperm(text_q["x"].size(1))[:int(text_q["x"].size(1) * self.codebook_prob)]
            # Make weight for q
            text_q_w = text_q["x"].new_zeros(text_q["x"].size(1))
            text_q_w[text_random_idx] = 1.0
            # Combine quantized codes and encoder output
            text_encoder_output["encoder_out"][0] = (
                text_q_w.view(-1, 1) * text_q["x"] + (- text_q_w + 1).view(-1, 1) * text_encoder_output["encoder_out"][0].transpose(0, 1)
            ).transpose(0, 1)

            text_codebook_out["prob_perplexity"] = text_q["prob_perplexity"]
            text_codebook_out["code_perplexity"] = text_q["code_perplexity"]
            text_codebook_out["num_vars"] = text_q["num_vars"]
            text_codebook_out["temp"] = text_q["temp"]

        masked_text_lm_results = (self.text_predictor(text_encoder_output["encoder_out"][0].transpose(0, 1), text_encoder_padding_mask), None)

        aux_target_masked_lm_results = None
        aux_target = speech_meta_info["aux_target"]
        if (aux_target is not None) and (speech_prenet_mode == 'policy_pretrain') and (self.update_num < 60000):
            mixed_source, unmasked_mixed_source, is_speech = self.interleave_and_mask_speech_text(
                discrete_batch['net_input']['unmasked_src_tokens'], 
                aux_target, self.interleave_ratio,
            )
            joint_encoder_speech_input = self.speech_encoder_prenet.encoder_prenet(mixed_source) 
            joint_encoder_text_input, _ = self.text_encoder_prenet(mixed_source)
            joint_encoder_input = is_speech.float()[..., None] * joint_encoder_speech_input + (~is_speech).float()[..., None] * joint_encoder_text_input

            # Encoder: T x B x C
            joint_encoder_output = self.encoder(joint_encoder_input, speech_encoder_padding_mask, tgt_layer=tgt_enc_layer)
            if self.use_codebook and self.training:
                joint_q = self.quantizer(joint_encoder_output["encoder_out"][0].transpose(0, 1))
                joint_random_idx = torch.randperm(joint_q['x'].size(1))[:int(joint_q['x'].size(1) * self.codebook_prob)]
                joint_q_w = joint_q['x'].new_zeros(joint_q['x'].size(1))
                joint_q_w[joint_random_idx] = 1.0
                joint_encoder_output['encoder_out'][0] = (
                    joint_q_w.view(-1, 1) * joint_q["x"] + (- joint_q_w + 1).view(-1, 1) * joint_encoder_output["encoder_out"][0].transpose(0, 1)
                ).transpose(0, 1)
           
            joint_encoder_speech_output = self.speech_predictor(joint_encoder_output['encoder_out'][0].transpose(0, 1), speech_encoder_padding_mask)
            joint_encoder_text_output = self.text_predictor(joint_encoder_output['encoder_out'][0].transpose(0, 1), speech_encoder_padding_mask)
            joint_encoder_output = is_speech.float()[..., None] * joint_encoder_speech_output + (~is_speech).float()[..., None] * joint_encoder_text_output

            mixed_meta_info = {
                'net_input': {
                    'src_tokens': mixed_source,
                    'unmasked_src_tokens': unmasked_mixed_source,
                },
                'target': unmasked_mixed_source,
            }
            aux_target_masked_lm_results = ((joint_encoder_output, None), mixed_meta_info)

        return None, None, ret_speech_codebook_outputs, text_codebook_out, ret_speech_masked_lm_results, masked_text_lm_results, ret_unigram_loss_speech, ret_unigram_loss_text, ret_speech_meta_info, aux_target_masked_lm_results

    def _integrate_with_speaker_cls(self, pad_input, encoder_input, encoder_padding_mask=None, cls_first=True):
        """
        encoder_input: [B, T, C]
        encoder_padding_mask: [B, T]
        """
        if hasattr(self, "text_decoder_prenet"):
            if isinstance(pad_input, tuple):
                repeat_cls_vector, repeat_cls_mask = pad_input
            else:
                repeat_cls_vector, repeat_cls_mask, _ = self.text_decoder_prenet(pad_input)

            if encoder_padding_mask is not None:
                bsz = encoder_input.size(0)
                tsz = encoder_input.size(1)
                encoder_padding_mask = encoder_input.new_zeros((bsz, tsz)) == 1.0
            if repeat_cls_mask is None:
                mask_size = (encoder_padding_mask.size(0), 1)
                mask_type = encoder_padding_mask.dtype
                repeat_cls_mask = encoder_padding_mask.new_zeros(mask_size) == 1.0
            ret_encoder_padding_mask = torch.cat([repeat_cls_mask, encoder_padding_mask], dim=1)

            if cls_first:
                ret_encoder_input = torch.cat([repeat_cls_vector, encoder_input], dim=1)
            else:
                ret_encoder_input = torch.cat([encoder_input, encoder_input[:,-1:,:]], dim=1)
                mask_size = (encoder_padding_mask.size(0), 1)
                mask_type = encoder_padding_mask.dtype
                repeat_cls_mask_ = encoder_padding_mask.new_ones(mask_size) == 1.0
                encoder_padding_mask_ = torch.cat([encoder_padding_mask, repeat_cls_mask_], dim=1)
                indices = encoder_padding_mask.eq(False).float().sum(1).type(torch.int64).unsqueeze(1)
                indices_mask = torch.zeros_like(ret_encoder_padding_mask).scatter(1, indices, 1.0)
                ret_encoder_input = ret_encoder_input * (1.0 - encoder_padding_mask_.type(ret_encoder_input.dtype).unsqueeze(2)) \
                    + repeat_cls_vector * indices_mask.type(repeat_cls_vector.dtype).unsqueeze(2)
            
        return ret_encoder_input, ret_encoder_padding_mask
     
    def get_baseline(self):
        return self.baseline[0]
    
    def update_baseline(self, reward):
        self.baseline[0] *= self.baseline_ema_coeff
        self.baseline[0] += ((1 - self.baseline_ema_coeff) * reward)

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        """NOT STRICT Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        # self.prune_modules(model_cfg.modules_filter)
        model_dict_size = self.text_decoder_postnet.output_projection.out_features
        ckpt_dict_size = state_dict["text_decoder_postnet.output_projection.weight"].size(0)
        if model_dict_size != ckpt_dict_size:
            # reset dictionary-related modules, such as embedding table and encoder ctc embed
            logger.warn(f"not equal dictionary between model and checkpoint: {model_dict_size} vs {ckpt_dict_size}")
            logger.info(f"reset model dictionary with size of {model_dict_size}")
            removed_keys = [
                key for key in state_dict.keys() if any(
                    key.startswith(previ) for previ in [
                        "encoder.proj", "text_encoder_prenet", "text_decoder_prenet", "text_decoder_postnet"
                    ]
                )
            ]
            for key in removed_keys:
                state_dict.pop(key, None)
                logger.info(f"removed loaded checkpoint: {key}")
        for m in self._modules.keys():
            m_state_dict = {
                key.replace(f"{m}.", ""): value for key, value in state_dict.items() if key.startswith(f"{m}.")
            }
            if hasattr(self, m):
                self._modules[m].load_state_dict(m_state_dict, False)
        return self

    def prune_modules(self, modules_filter=None):
        """Prune unused modules for specific tasks."""
        if modules_filter is None:
            return
        elif modules_filter == "s2c":
            if hasattr(self, "text_encoder_prenet"): del self.text_encoder_prenet
            if hasattr(self, "speech_decoder_prenet") and getattr(self.args, "sid_pooling_layer", None) != "decoder-las": 
                del self.speech_decoder_prenet
            if hasattr(self, "speech_decoder_postnet"): del self.speech_decoder_postnet
            if hasattr(self, "text_decoder_postnet"): del self.text_decoder_postnet
            if hasattr(self, "speech_encoder_postnet"): del self.speech_encoder_postnet
            if hasattr(self.encoder, "proj"): self.encoder.proj = None
            if hasattr(self, "projection"): del self.projection
            if hasattr(self, "quantizer"): del self.quantizer
            if getattr(self.args, "sid_pooling_layer", "decoder").startswith("encoder") or getattr(self.args, "sid_decoder_speaker", False): 
                if hasattr(self.decoder, "dropout_module"): del self.decoder.dropout_module
                if hasattr(self.decoder, "layers"): del self.decoder.layers
                if hasattr(self.decoder, "layer_norm"): del self.decoder.layer_norm
                if hasattr(self, "text_decoder_prenet"): del self.text_decoder_prenet
        elif modules_filter == "s2s":
            if hasattr(self, "speaker_decoder_postnet"): del self.speaker_decoder_postnet
            if hasattr(self, "text_encoder_prenet"): del self.text_encoder_prenet
            if hasattr(self, "text_decoder_prenet"): del self.text_decoder_prenet
            if hasattr(self, "text_decoder_postnet"): del self.text_decoder_postnet
            if hasattr(self, "speech_encoder_postnet"): del self.speech_encoder_postnet
            if hasattr(self.encoder, "proj"): self.encoder.proj = None
            if hasattr(self, "projection"): del self.projection
            if hasattr(self, "quantizer"): del self.quantizer
        elif modules_filter == "t2s":
            if hasattr(self, "speaker_decoder_postnet"): del self.speaker_decoder_postnet
            if hasattr(self, "speech_encoder_prenet"): del self.speech_encoder_prenet
            if hasattr(self, "text_decoder_prenet"): del self.text_decoder_prenet
            if hasattr(self, "text_decoder_postnet"): del self.text_decoder_postnet
            if hasattr(self, "speech_encoder_postnet"): del self.speech_encoder_postnet
            if hasattr(self.encoder, "proj"): self.encoder.proj = None
            if hasattr(self, "projection"): del self.projection
            if hasattr(self, "quantizer"): del self.quantizer
        elif modules_filter == "s3prl":
            # remain the encoder and the pre/post net
            if hasattr(self.decoder, "dropout_module"): del self.decoder.dropout_module
            if hasattr(self.decoder, "layers"): del self.decoder.layers
            if hasattr(self.decoder, "layer_norm"): del self.decoder.layer_norm
            if hasattr(self, "speaker_decoder_postnet"): del self.speaker_decoder_postnet
            if hasattr(self, "text_decoder_prenet"): del self.text_decoder_prenet
            if hasattr(self, "text_decoder_postnet"): del self.text_decoder_postnet
            if hasattr(self, "speech_decoder_prenet"): del self.speech_decoder_prenet
            if hasattr(self, "speech_decoder_postnet"): del self.speech_decoder_postnet
            if hasattr(self, "speech_encoder_postnet"): del self.speech_encoder_postnet
            if hasattr(self.encoder, "proj"): self.encoder.proj = None
            if hasattr(self, "projection"): del self.projection
            if hasattr(self, "quantizer"): del self.quantizer

    def forward_encoder_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward_encoder(
                src_tokens=net_input["src_tokens"],
            )
        else:
            return self.forward_encoder_non_torchscript(net_input)

    @torch.jit.unused
    def forward_encoder_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens" and k != "task_name" and k != "src_lengths"
        }
        return self.forward_encoder(**encoder_input)

    def forward_encoder(self, src_tokens):
        # Encoder Prenet
        encoder_input, encoder_padding_mask = self.speech_encoder_prenet(src_tokens)

        # Encoder
        encoder_output = self.encoder(encoder_input, encoder_padding_mask)

        return encoder_output

    def forward_text_encoder(self, src_tokens):
        # Text Encoder Prenet
        encoder_input, encoder_padding_mask = self.text_encoder_prenet(src_tokens)

        # Encoder
        encoder_output = self.encoder(encoder_input, encoder_padding_mask)

        return encoder_output

    def forward_decoder(self, tokens, encoder_out, incremental_state):
        # Decoder Prenet
        prev_output_tokens, tgt_mask, incremental_state = self.text_decoder_prenet(tokens, incremental_state)

        # Decoder
        decoder_output, extra = self.decoder(
            prev_output_tokens,
            tgt_mask,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )

        # Decoder Postnet
        return self.text_decoder_postnet(decoder_output), extra

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def generate_speech(self, source=None, src_tokens=None, spkembs=None, **kwargs):
        assert source is not None or src_tokens is not None

        threshold = kwargs.get("threshold", 0.5)
        minlenratio = kwargs.get("threshold", 0.0)

        if source is None:
            assert src_tokens.size(0) == 1
            encoder_out = self.forward_text_encoder(src_tokens)
            maxlenratio = kwargs.get("threshold", 20.0)
        else:
            assert source.size(0) == 1
            encoder_out = self.forward_encoder(source, padding_mask=kwargs["padding_mask"])
            maxlenratio = kwargs.get("threshold", 10.0)

        if spkembs is not None and self.spk_embed_integration_type != "pre":
            encoder_out["encoder_out"] = [self._integrate_with_spk_embed(
                encoder_out["encoder_out"][0].transpose(0, 1), spkembs
            ).transpose(0, 1)]
            spkembs = None

        maxlen = int(encoder_out["encoder_out"][0].size(0) * maxlenratio / self.reduction_factor)
        minlen = int(encoder_out["encoder_out"][0].size(0) * minlenratio / self.reduction_factor)
        
        idx = 0
        ys = encoder_out["encoder_out"][0].new_zeros(1, 1, self.speech_decoder_postnet.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        if isinstance(self.decoder, FairseqIncrementalDecoder):
            incremental_states = {}
        else:
            incremental_states = None
        attns = []
        while True:
            # update index
            idx += 1
            # calculate output and stop prob at idx-th step
            decoder_in, _ = self.speech_decoder_prenet(ys, spkembs=spkembs)
            z, extra = self.decoder(decoder_in[:,-1:], None, encoder_out, incremental_states, alignment_layer=-1)
            outs += [self.speech_decoder_postnet.feat_out(z[0, -1]).view(self.reduction_factor, self.speech_decoder_postnet.odim)]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.speech_decoder_postnet.prob_out(z[0, -1]))]  # [(r), ...]

            # update next inputs
            ys = torch.cat((ys, outs[-1][-1].view(1, 1, self.speech_decoder_postnet.odim)), dim=1)  # (1, idx + 1, odim)
            attns.append(torch.stack([att_l[0] for att_l in extra['attn'][0]], dim=0))
            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = (torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2))  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.speech_decoder_postnet.postnet is not None:
                    outs = outs + self.speech_decoder_postnet.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                attn = torch.cat(attns, dim=2)
                break

        if outs.size(0) == maxlen:
            logging.warning("output length reaches maximum length")
        return outs, probs, attn


@register_model_architecture(model_name="t5_transformer", arch_name="t5_transformer")
def base_architecture(args):
    # Transformer
    args.bert_init = getattr(args, "bert_init", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.max_text_positions = getattr(args, "max_text_positions", DEFAULT_MAX_TEXT_POSITIONS)
    args.max_speech_positions = getattr(args, "max_speech_positions", DEFAULT_MAX_SPEECH_POSITIONS)

    # Espnet related, including prenet, postnet
    args.eprenet_conv_layers = getattr(args, "eprenet_conv_layers", 0)
    args.eprenet_conv_filts = getattr(args, "eprenet_conv_filts", 0)
    args.eprenet_conv_chans = getattr(args, "eprenet_conv_chans", 0)
    args.use_batch_norm = getattr(args, "use_batch_norm", True)
    args.eprenet_dropout_rate = getattr(args, "eprenet_dropout_rate", 0.0)
    args.enc_use_scaled_pos_enc = getattr(args, "enc_use_scaled_pos_enc", True)
    args.dec_use_scaled_pos_enc = getattr(args, "dec_use_scaled_pos_enc", True)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_chans = getattr(args, "postnet_chans", 256)
    args.postnet_filts = getattr(args, "postnet_filts", 5)
    args.postnet_dropout_rate = getattr(args, "postnet_dropout_rate", 0.5)
    args.dprenet_dropout_rate = getattr(args, "dprenet_dropout_rate", 0.5)
    args.dprenet_layers = getattr(args, "dprenet_layers", 2)
    args.dprenet_units = getattr(args, "dprenet_units", 256)
    args.initial_encoder_alpha = getattr(args, "initial_encoder_alpha", 1.0)
    args.initial_decoder_alpha = getattr(args, "initial_decoder_alpha", 1.0)
    args.spk_embed_integration_type = getattr(args, "spk_embed_integration_type", "pre")
    args.spk_embed_dim = getattr(args, "spk_embed_dim", 256)
    args.encoder_reduction_factor = getattr(args, "encoder_reduction_factor", 1)
    args.reduction_factor = getattr(args, "reduction_factor", 1)
    args.transformer_enc_positional_dropout_rate = getattr(args, "transformer_enc_positional_dropout_rate", 0.1)
    args.transformer_dec_positional_dropout_rate = getattr(args, "transformer_dec_positional_dropout_rate", 0.1)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-5)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    # Convolutional subsampler
    args.encoder_speech_prenet = getattr(args, "encoder_speech_prenet", "linear")
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.share_input_output_embed = getattr(args, "share_input_output_embed", False)
    args.share_ctc_embed = getattr(args, "share_ctc_embed", False)
    args.freeze_encoder_updates = getattr(args, "freeze_encoder_updates", 0)
    args.freeze_decoder_updates = getattr(args, "freeze_decoder_updates", 0)
    args.no_freeze_encoder_layer = getattr(args, "no_freeze_encoder_layer", None)

    ## sid
    args.sid_embed_dim = getattr(args, "sid_embed_dim", 128)
    args.sid_pooling_layer = getattr(args, "sid_pooling_layer", "decoder")
    args.softmax_scale = getattr(args, "softmax_scale", 1)
    args.softmax_margin = getattr(args, "softmax_margin", 0)
    args.softmax_easy_margin = getattr(args, "softmax_easy_margin", False)
    args.modules_filter = getattr(args, "modules_filter", None)

    ## Hubert
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    args.target_glu = getattr(args, "target_glu", False)
    args.logit_temp = getattr(args, "logit_temp", 0.1)
    args.final_dim = getattr(args, "final_dim", 256)
    args.untie_final_proj = getattr(args, "untie_final_proj", True)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.1)
    args.use_sent_enc_layer = getattr(args, "use_sent_enc_layer", True)
    # hubert feature extractor
    args.extractor_mode = getattr(args, "extractor_mode", "default")
    args.conv_feature_layers = getattr(args, "conv_feature_layers", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
    args.conv_bias = getattr(args, "conv_bias", False)
    # mask
    args.hubert_mask_length = getattr(args, "hubert_mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.0)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)
    # channel mask
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)
    # loss computation
    args.skip_masked = getattr(args, "skip_masked", False)
    args.skip_nomask = getattr(args, "skip_nomask", False)
    # conv Pos
    args.use_conv_pos = getattr(args, "use_conv_pos", False)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", False)

    # codebook
    args.use_codebook = getattr(args, "use_codebook", False)
    args.latent_vars = getattr(args, "latent_vars", 300)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)
    args.latent_temp = getattr(args, "latent_temp", (2, 0.5, 0.999995))
    args.quantizer_depth = getattr(args, "quantizer_depth", 1)
    args.quantizer_factor = getattr(args, "quantizer_factor", 3)
    args.codebook_prob = getattr(args, "codebook_prob", 0.5)
    args.aux_target_loss_weight = getattr(args, "aux_target_loss_weight", 1)

    # Relative pos embed
    args.relative_position_embedding = getattr(args, "relative_position_embedding", False)
    args.num_buckets = getattr(args, "num_buckets", 320)
    args.max_distance = getattr(args, "max_distance", 1280)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 160)
    args.decoder_max_relative_position = getattr(args, "decoder_max_relative_position", 160)
    
    # gan related parameters
    args.start_gan_updates = getattr(args, "start_gan_updates", -1)
    args.discriminator_dim = getattr(args,"discriminator_dim", 768)
    args.discriminator_kernel = getattr(args, "discriminator_kernel", 8)
    args.discriminator_dilation = getattr(args, "discriminator_dilation", 1)
    args.discriminator_max_pool = getattr(args, "discriminator_max_pool", False)
    args.discriminator = getattr(args, "discriminator_causal", True)
    args.discriminator_spectral_norm = getattr(args, "discriminator_spectral_norm", False)
    args.discriminator_weight_norm = getattr(args, "discriminator_weight_norm", False)
    args.discriminator_dropout = getattr(args, "discriminator_dropout", 0.0)
    args.discriminator_depth = getattr(args, "discriminator_depth", 3)
    args.discriminator_linear_emb = getattr(args, "discriminator_linear_emb", False)
    args.discriminator_act_after_linear = getattr(args, "discriminator_act_after_linear", False)
    args.policy_feat_dim = getattr(args, "policy_feat_dim", 768)
    args.no_load_policy = getattr(args, "no_load_policy", False)
    args.no_load_speech_prepost = getattr(args, "no_load_speech_prepost", False)
    args.no_load_text_prepost = getattr(args, "no_load_text_prepost", False)
    args.no_load_quantizer = getattr(args, "no_load_quantizer", False)
    args.use_transformer_policy = getattr(args, "use_transformer_policy", False)
    args.use_softmax_soft_pool  = getattr(args, "use_softmax_soft_pool", False)
    args.speech_encoder_prenet_mask = getattr(args, "speech_encoder_prenet_mask", 0.3)
    args.speech_encoder_prenet_mask_random = getattr(args, "speech_encoder_prenet_mask_random", 0.1)
    args.speech_encoder_prenet_mask_length = getattr(args, "speech_encoder_prenet_mask_length", "span-poisson")
    args.speech_encoder_prenet_poisson_lambda = getattr(args, "speech_encoder_prenet_poisson_lambda", 3.5)
    args.speech_encoder_prenet_replace_length = getattr(args, "speech_encoder_prenet_replace_length", 1)
    args.speech_encoder_prenet_iid_noise_target = getattr(args, "speech_encoder_prenet_iid_noise_target", False)
    args.word_freq = getattr(args, "word_freq", 12)

    args.turn_off_finetune_speech_enc_dec = getattr(args, "turn_off_finetune_speech_enc_dec", False) 
    args.turn_off_finetune_all_enc_dec = getattr(args, "turn_off_finetune_all_enc_dec", False)
    
    args.km_size = getattr(args, "km_size", 1025)
    args.frame_target_classes = getattr(args, "frame_target_classes", 100)
    args.init_clus_dir = getattr(args, "init_clus_dir")
    args.init_model_dir = getattr(args, "init_model_dir", None)
    args.sampling_nums = getattr(args, "sampling_nums", 3)
    args.baseline_ema_coeff = getattr(args, "baseline_ema_coeff", 0.99)

@register_model_architecture("t5_transformer", "t5_transformer_base")
def t5_transformer_base(args):
    args.use_conv_pos = getattr(args, "use_conv_pos", True)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    args.mask_prob = getattr(args, "mask_prob", 0.80)
    base_architecture(args)

@register_model_architecture("t5_transformer", "t5_transformer_large")
def t5_transformer_large(args):
    args.use_conv_pos = getattr(args, "use_conv_pos", True)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.layer_norm_first = getattr(args, "layer_norm_first", True)
    args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)
    args.extractor_mode = getattr(args, "extractor_mode", "layer_norm")
    args.final_dim = getattr(args, "final_dim", 768)
    args.mask_prob = getattr(args, "mask_prob", 0.80)
    base_architecture(args)

@register_model_architecture("t5_transformer", "t5_transformer_base_asr")
def t5_transformer_base_asr(args):
    args.use_conv_pos = getattr(args, "use_conv_pos", True)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.0)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.1)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.1)
    args.mask_prob = getattr(args, "mask_prob", 0.75)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_channel_length = getattr(args, "mask_channel_length", 64)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.max_text_positions = getattr(args, "max_text_positions", 600)
    base_architecture(args)


@register_model_architecture(model_name="t5_transformer", arch_name="t5_transformer_1_layer")
def base_1_layer_architecture(args):
    # Transformer
    args.bert_init = getattr(args, "bert_init", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.max_text_positions = getattr(args, "max_text_positions", DEFAULT_MAX_TEXT_POSITIONS)
    args.max_speech_positions = getattr(args, "max_speech_positions", DEFAULT_MAX_SPEECH_POSITIONS)
    args.aux_target_loss_weight = getattr(args, "aux_target_loss_weight", 1)

    # Espnet related, including prenet, postnet
    args.eprenet_conv_layers = getattr(args, "eprenet_conv_layers", 0)
    args.eprenet_conv_filts = getattr(args, "eprenet_conv_filts", 0)
    args.eprenet_conv_chans = getattr(args, "eprenet_conv_chans", 0)
    args.use_batch_norm = getattr(args, "use_batch_norm", True)
    args.eprenet_dropout_rate = getattr(args, "eprenet_dropout_rate", 0.0)
    args.enc_use_scaled_pos_enc = getattr(args, "enc_use_scaled_pos_enc", True)
    args.dec_use_scaled_pos_enc = getattr(args, "dec_use_scaled_pos_enc", True)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_chans = getattr(args, "postnet_chans", 256)
    args.postnet_filts = getattr(args, "postnet_filts", 5)
    args.postnet_dropout_rate = getattr(args, "postnet_dropout_rate", 0.5)
    args.dprenet_dropout_rate = getattr(args, "dprenet_dropout_rate", 0.5)
    args.dprenet_layers = getattr(args, "dprenet_layers", 2)
    args.dprenet_units = getattr(args, "dprenet_units", 256)
    args.initial_encoder_alpha = getattr(args, "initial_encoder_alpha", 1.0)
    args.initial_decoder_alpha = getattr(args, "initial_decoder_alpha", 1.0)
    args.spk_embed_integration_type = getattr(args, "spk_embed_integration_type", "pre")
    args.spk_embed_dim = getattr(args, "spk_embed_dim", 256)
    args.encoder_reduction_factor = getattr(args, "encoder_reduction_factor", 1)
    args.reduction_factor = getattr(args, "reduction_factor", 1)
    args.transformer_enc_positional_dropout_rate = getattr(args, "transformer_enc_positional_dropout_rate", 0.1)
    args.transformer_dec_positional_dropout_rate = getattr(args, "transformer_dec_positional_dropout_rate", 0.1)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-5)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    # Convolutional subsampler
    args.encoder_speech_prenet = getattr(args, "encoder_speech_prenet", "linear")
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.share_input_output_embed = getattr(args, "share_input_output_embed", False)
    args.share_ctc_embed = getattr(args, "share_ctc_embed", False)
    args.freeze_encoder_updates = getattr(args, "freeze_encoder_updates", 0)
    args.freeze_decoder_updates = getattr(args, "freeze_decoder_updates", 0)
    args.no_freeze_encoder_layer = getattr(args, "no_freeze_encoder_layer", None)

    ## sid
    args.sid_embed_dim = getattr(args, "sid_embed_dim", 128)
    args.sid_pooling_layer = getattr(args, "sid_pooling_layer", "decoder")
    args.softmax_scale = getattr(args, "softmax_scale", 1)
    args.softmax_margin = getattr(args, "softmax_margin", 0)
    args.softmax_easy_margin = getattr(args, "softmax_easy_margin", False)
    args.modules_filter = getattr(args, "modules_filter", None)

    ## Hubert
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    args.target_glu = getattr(args, "target_glu", False)
    args.logit_temp = getattr(args, "logit_temp", 0.1)
    args.final_dim = getattr(args, "final_dim", 256)
    args.untie_final_proj = getattr(args, "untie_final_proj", True)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.1)
    args.use_sent_enc_layer = getattr(args, "use_sent_enc_layer", True)
    # hubert feature extractor
    args.extractor_mode = getattr(args, "extractor_mode", "default")
    args.conv_feature_layers = getattr(args, "conv_feature_layers", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
    args.conv_bias = getattr(args, "conv_bias", False)
    # mask
    args.hubert_mask_length = getattr(args, "hubert_mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.0)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)
    # channel mask
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)
    # loss computation
    args.skip_masked = getattr(args, "skip_masked", False)
    args.skip_nomask = getattr(args, "skip_nomask", False)
    # conv Pos
    args.use_conv_pos = getattr(args, "use_conv_pos", False)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", False)

    # codebook
    args.use_codebook = getattr(args, "use_codebook", False)
    args.latent_vars = getattr(args, "latent_vars", 300)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)
    args.latent_temp = getattr(args, "latent_temp", (2, 0.5, 0.999995))
    args.quantizer_depth = getattr(args, "quantizer_depth", 1)
    args.quantizer_factor = getattr(args, "quantizer_factor", 3)
    args.codebook_prob = getattr(args, "codebook_prob", 0.5)

    # Relative pos embed
    args.relative_position_embedding = getattr(args, "relative_position_embedding", False)
    args.num_buckets = getattr(args, "num_buckets", 320)
    args.max_distance = getattr(args, "max_distance", 1280)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 160)
    args.decoder_max_relative_position = getattr(args, "decoder_max_relative_position", 160)
    
    # gan related parameters
    args.start_gan_updates = getattr(args, "start_gan_updates", -1)
    args.discriminator_dim = getattr(args,"discriminator_dim", 768)
    args.discriminator_kernel = getattr(args, "discriminator_kernel", 8)
    args.discriminator_dilation = getattr(args, "discriminator_dilation", 1)
    args.discriminator_max_pool = getattr(args, "discriminator_max_pool", False)
    args.discriminator = getattr(args, "discriminator_causal", True)
    args.discriminator_spectral_norm = getattr(args, "discriminator_spectral_norm", False)
    args.discriminator_weight_norm = getattr(args, "discriminator_weight_norm", False)
    args.discriminator_dropout = getattr(args, "discriminator_dropout", 0.0)
    args.discriminator_depth = getattr(args, "discriminator_depth", 3)
    args.discriminator_linear_emb = getattr(args, "discriminator_linear_emb", False)
    args.discriminator_act_after_linear = getattr(args, "discriminator_act_after_linear", False)
    args.policy_feat_dim = getattr(args, "policy_feat_dim", 768)
    args.no_load_policy = getattr(args, "no_load_policy", False)
    args.no_load_speech_prepost = getattr(args, "no_load_speech_prepost", False)
    args.no_load_text_prepost = getattr(args, "no_load_text_prepost", False)
    args.no_load_quantizer = getattr(args, "no_load_quantizer", False)
    args.use_transformer_policy = getattr(args, "use_transformer_policy", False)
    args.use_softmax_soft_pool  = getattr(args, "use_softmax_soft_pool", False)
    args.speech_encoder_prenet_mask = getattr(args, "speech_encoder_prenet_mask", 0.3)
    args.speech_encoder_prenet_mask_random = getattr(args, "speech_encoder_prenet_mask_random", 0.1)
    args.speech_encoder_prenet_mask_length = getattr(args, "speech_encoder_prenet_mask_length", "span-poisson")
    args.speech_encoder_prenet_poisson_lambda = getattr(args, "speech_encoder_prenet_poisson_lambda", 3.5)
    args.speech_encoder_prenet_replace_length = getattr(args, "speech_encoder_prenet_replace_length", 1)
    args.speech_encoder_prenet_iid_noise_target = getattr(args, "speech_encoder_prenet_iid_noise_target", False)
    args.word_freq = getattr(args, "word_freq", 12)
    
    args.turn_off_finetune_speech_enc_dec = getattr(args, "turn_off_finetune_speech_enc_dec", False) 
    args.turn_off_finetune_all_enc_dec = getattr(args, "turn_off_finetune_all_enc_dec", False)
    
    args.km_size = getattr(args, "km_size", 1025)
    args.frame_target_classes = getattr(args, "frame_target_classes", 100)
    args.init_clus_dir = getattr(args, "init_clus_dir")
    args.init_model_dir = getattr(args, "init_model_dir", None)
    args.sampling_nums = getattr(args, "sampling_nums", 3)
    args.baseline_ema_coeff = getattr(args, "baseline_ema_coeff", 0.99)
    
    
@register_model_architecture(model_name="t5_transformer", arch_name="t5_transformer_1_layer_384")
def base_1_layer_architecture_384(args):
    # Transformer
    args.bert_init = getattr(args, "bert_init", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim",384)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 384 * 4)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 6)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 6)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.max_text_positions = getattr(args, "max_text_positions", DEFAULT_MAX_TEXT_POSITIONS)
    args.max_speech_positions = getattr(args, "max_speech_positions", DEFAULT_MAX_SPEECH_POSITIONS)

    # Espnet related, including prenet, postnet
    args.eprenet_conv_layers = getattr(args, "eprenet_conv_layers", 0)
    args.eprenet_conv_filts = getattr(args, "eprenet_conv_filts", 0)
    args.eprenet_conv_chans = getattr(args, "eprenet_conv_chans", 0)
    args.use_batch_norm = getattr(args, "use_batch_norm", True)
    args.eprenet_dropout_rate = getattr(args, "eprenet_dropout_rate", 0.0)
    args.enc_use_scaled_pos_enc = getattr(args, "enc_use_scaled_pos_enc", True)
    args.dec_use_scaled_pos_enc = getattr(args, "dec_use_scaled_pos_enc", True)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_chans = getattr(args, "postnet_chans", 256)
    args.postnet_filts = getattr(args, "postnet_filts", 5)
    args.postnet_dropout_rate = getattr(args, "postnet_dropout_rate", 0.5)
    args.dprenet_dropout_rate = getattr(args, "dprenet_dropout_rate", 0.5)
    args.dprenet_layers = getattr(args, "dprenet_layers", 2)
    args.dprenet_units = getattr(args, "dprenet_units", 256)
    args.initial_encoder_alpha = getattr(args, "initial_encoder_alpha", 1.0)
    args.initial_decoder_alpha = getattr(args, "initial_decoder_alpha", 1.0)
    args.spk_embed_integration_type = getattr(args, "spk_embed_integration_type", "pre")
    args.spk_embed_dim = getattr(args, "spk_embed_dim", 256)
    args.encoder_reduction_factor = getattr(args, "encoder_reduction_factor", 1)
    args.reduction_factor = getattr(args, "reduction_factor", 1)
    args.transformer_enc_positional_dropout_rate = getattr(args, "transformer_enc_positional_dropout_rate", 0.1)
    args.transformer_dec_positional_dropout_rate = getattr(args, "transformer_dec_positional_dropout_rate", 0.1)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-5)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    # Convolutional subsampler
    args.encoder_speech_prenet = getattr(args, "encoder_speech_prenet", "linear")
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.share_input_output_embed = getattr(args, "share_input_output_embed", False)
    args.share_ctc_embed = getattr(args, "share_ctc_embed", False)
    args.freeze_encoder_updates = getattr(args, "freeze_encoder_updates", 0)
    args.freeze_decoder_updates = getattr(args, "freeze_decoder_updates", 0)
    args.no_freeze_encoder_layer = getattr(args, "no_freeze_encoder_layer", None)

    ## sid
    args.sid_embed_dim = getattr(args, "sid_embed_dim", 128)
    args.sid_pooling_layer = getattr(args, "sid_pooling_layer", "decoder")
    args.softmax_scale = getattr(args, "softmax_scale", 1)
    args.softmax_margin = getattr(args, "softmax_margin", 0)
    args.softmax_easy_margin = getattr(args, "softmax_easy_margin", False)
    args.modules_filter = getattr(args, "modules_filter", None)

    ## Hubert
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    args.target_glu = getattr(args, "target_glu", False)
    args.logit_temp = getattr(args, "logit_temp", 0.1)
    args.final_dim = getattr(args, "final_dim", 256)
    args.untie_final_proj = getattr(args, "untie_final_proj", True)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.1)
    args.use_sent_enc_layer = getattr(args, "use_sent_enc_layer", True)
    # hubert feature extractor
    args.extractor_mode = getattr(args, "extractor_mode", "default")
    args.conv_feature_layers = getattr(args, "conv_feature_layers", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
    args.conv_bias = getattr(args, "conv_bias", False)
    # mask
    args.hubert_mask_length = getattr(args, "hubert_mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.0)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)
    # channel mask
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)
    # loss computation
    args.skip_masked = getattr(args, "skip_masked", False)
    args.skip_nomask = getattr(args, "skip_nomask", False)
    # conv Pos
    args.use_conv_pos = getattr(args, "use_conv_pos", False)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", False)

    # codebook
    args.use_codebook = getattr(args, "use_codebook", False)
    args.latent_vars = getattr(args, "latent_vars", 300)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)
    args.latent_temp = getattr(args, "latent_temp", (2, 0.5, 0.999995))
    args.quantizer_depth = getattr(args, "quantizer_depth", 1)
    args.quantizer_factor = getattr(args, "quantizer_factor", 3)
    args.codebook_prob = getattr(args, "codebook_prob", 0.5)

    # Relative pos embed
    args.relative_position_embedding = getattr(args, "relative_position_embedding", False)
    args.num_buckets = getattr(args, "num_buckets", 320)
    args.max_distance = getattr(args, "max_distance", 1280)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 160)
    args.decoder_max_relative_position = getattr(args, "decoder_max_relative_position", 160)
    
    # gan related parameters
    args.start_gan_updates = getattr(args, "start_gan_updates", -1)
    args.discriminator_dim = getattr(args,"discriminator_dim", 384)
    args.discriminator_kernel = getattr(args, "discriminator_kernel", 8)
    args.discriminator_dilation = getattr(args, "discriminator_dilation", 1)
    args.discriminator_max_pool = getattr(args, "discriminator_max_pool", False)
    args.discriminator = getattr(args, "discriminator_causal", True)
    args.discriminator_spectral_norm = getattr(args, "discriminator_spectral_norm", False)
    args.discriminator_weight_norm = getattr(args, "discriminator_weight_norm", False)
    args.discriminator_dropout = getattr(args, "discriminator_dropout", 0.0)
    args.discriminator_depth = getattr(args, "discriminator_depth", 3)
    args.discriminator_linear_emb = getattr(args, "discriminator_linear_emb", False)
    args.discriminator_act_after_linear = getattr(args, "discriminator_act_after_linear", False)
    args.policy_feat_dim = getattr(args, "policy_feat_dim", 768)
    args.no_load_policy = getattr(args, "no_load_policy", False)
    args.no_load_speech_prepost = getattr(args, "no_load_speech_prepost", False)
    args.no_load_text_prepost = getattr(args, "no_load_text_prepost", False)
    args.no_load_quantizer = getattr(args, "no_load_quantizer", False)
    args.use_transformer_policy = getattr(args, "use_transformer_policy", False)
    args.use_softmax_soft_pool  = getattr(args, "use_softmax_soft_pool", False)
    args.speech_encoder_prenet_mask = getattr(args, "speech_encoder_prenet_mask", 0.3)
    args.speech_encoder_prenet_mask_random = getattr(args, "speech_encoder_prenet_mask_random", 0.1)
    args.speech_encoder_prenet_mask_length = getattr(args, "speech_encoder_prenet_mask_length", "span-poisson")
    args.speech_encoder_prenet_poisson_lambda = getattr(args, "speech_encoder_prenet_poisson_lambda", 3.5)
    args.speech_encoder_prenet_replace_length = getattr(args, "speech_encoder_prenet_replace_length", 1)
    args.speech_encoder_prenet_iid_noise_target = getattr(args, "speech_encoder_prenet_iid_noise_target", False)
    args.word_freq = getattr(args, "word_freq", 12)
    
    args.turn_off_finetune_speech_enc_dec = getattr(args, "turn_off_finetune_speech_enc_dec", False) 
    args.turn_off_finetune_all_enc_dec = getattr(args, "turn_off_finetune_all_enc_dec", False)
    
    args.km_size = getattr(args, "km_size", 1025)
    args.frame_target_classes = getattr(args, "frame_target_classes", 100)
    args.init_clus_dir = getattr(args, "init_clus_dir")
    args.init_model_dir = getattr(args, "init_model_dir", None)
    args.sampling_nums = getattr(args, "sampling_nums", 3)
    args.baseline_ema_coeff = getattr(args, "baseline_ema_coeff", 0.99)
    

@register_model_architecture("t5_transformer", "t5_transformer_base_1_layer")
def t5_transformer_base_1_layer(args):
    args.use_conv_pos = getattr(args, "use_conv_pos", True)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    args.mask_prob = getattr(args, "mask_prob", 0.80)
    base_1_layer_architecture(args)
    
    
@register_model_architecture("t5_transformer", "t5_transformer_base_1_layer_384")
def t5_transformer_base_1_layer_384(args):
    args.use_conv_pos = getattr(args, "use_conv_pos", True)
    args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    args.mask_prob = getattr(args, "mask_prob", 0.80)
    base_1_layer_architecture_384(args)

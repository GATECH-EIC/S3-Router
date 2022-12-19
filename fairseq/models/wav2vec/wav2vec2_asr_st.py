# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import math
import re
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from fairseq.models.wav2vec.operation_st import Linear_ST, Linear_ST_InStr, Linear_ST_OutStr, Conv1d_ST

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

import re


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint_activations"}
    )
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")


@dataclass
class Wav2Vec2CtcSTConfig(Wav2Vec2AsrConfig):
    blank_weight: float = 0
    blank_mode: str = "add"

    prune_rate: float = field(
        default=0.5,
        metadata={
            "help": "weight remaining ratio"
        },
    )

    no_prune_proj: bool = field(
        default=True,
        metadata={
            "help": "do not prune the final projection layer"
        },
    )

    no_prune_extractor: bool = field(
        default=True,
        metadata={
            "help": "do not prune the conv-based feature extractor"
        },
    )

    no_prune_pos: bool = field(
        default=True,
        metadata={
            "help": "do not prune the position embedding layer"
        },
    )

    no_prune_post_extract_proj: bool = field(
        default=False,
        metadata={
            "help": "do not prune the post extractor projection layer"
        },
    )

    trainable_bias: bool = field(
        default=False,
        metadata={
            "help": "train the bias"
        },
    )

    trainable_layer_norm: bool = field(
        default=False,
        metadata={
            "help": "train the layer norm"
        },
    )

    trainable_if_unpruned: bool = field(
        default=False,
        metadata={
            "help": "train all the other params where masks are not applied"
        },
    )

    trainable_pos: bool = field(
        default=False,
        metadata={
            "help": "trainable positional conv"
        },
    )

    trainable_proj: bool = field(
        default=False,
        metadata={
            "help": "trainable final projection layer"
        },
    )

    trainable_post_extract_proj: bool = field(
        default=False,
        metadata={
            "help": "trainable post extraction projection layer"
        },
    )

    init_score: str = field(
        default='kaiming',
        metadata={
            "help": "score initialization"
        },
    )

    init_score_scale: float = field(
        default=1.,
        metadata={
            "help": "scale for init_score_with_weight_mag_with_scale"
        },
    )

    fix_attn: bool = field(
        default=False,
        metadata={
            "help": "fix the mask on the self-attention layers"
        },
    )

    fix_fc: bool = field(
        default=False,
        metadata={
            "help": "fix the mask on the FC layers"
        },
    )

    layerwise_prune_rate: Optional[Tuple[float, float, float, float, float, float, float, float, float, float, float, float]] = field(
        default=None,
        metadata={
            "help": (
                "layerwise prune rate"
            )
        },
    )

    fc_attn_prune_rate: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "help": (
                "prune rate for fc and self-attn layers"
            )
        },
    )

    co_train: bool = field(
        default=False,
        metadata={
            "help": "co-train the model weights and the mask"
        },
    )

    ste_sigmoid: bool = field(
        default=False,
        metadata={
            "help": "use sigmoid in STE for updating masks"
        },
    )

    linear_st_op: str = field(
        default='linear_st',
        metadata={
            "help": "sparse operation for replacing linear layers"
        },
    )

    fix_mask: bool = field(
        default=False,
        metadata={
            "help": "whether to fix all the masks"
        },
    )

    fix_mask_before: int = field(
        default=0,
        metadata={
            "help": "fix all the masks before the given layer id"
        },
    )

    load_mask_only: bool = field(
        default=False,
        metadata={
            "help": "whether to only load the masks from the pretrained model"
        },
    )

    load_mask_before: int = field(
        default=0,
        metadata={
            "help": "load all the masks before the given layer id"
        },
    )

    eval_only: bool = field(
        default=False,
        metadata={
            "help": "do evaluation only"
        },
    )


@register_model("wav2vec_ctc_st", dataclass=Wav2Vec2CtcSTConfig)
class Wav2VecCtcST(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcSTConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcSTConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))

        if not cfg.co_train:
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "weight") and m.weight is not None:
                    if not hasattr(m, "set_prune_rate"):  ## For some unpruned and trainable weights, do not stop their gradients
                        if cfg.trainable_if_unpruned:
                            continue
                        if cfg.trainable_pos and 'pos_conv' in n:
                            continue
                        if cfg.trainable_proj and 'proj' in n and 'post' not in n:
                            continue
                        if cfg.trainable_post_extract_proj and 'post_extract_proj' in n:
                            continue
                        if cfg.trainable_layer_norm and 'layer_norm' in n:
                            continue

                    if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
                        print(f"==> No gradient to {n}.weight_g and {n}.weight_v")

                        m.weight_g.requires_grad = False
                        m.weight_v.requires_grad = False

                        if m.weight_g.grad is not None:
                            print(f"==> Setting gradient of {n}.weight_g to None")
                            m.weight_g.grad = None

                        if m.weight_v.grad is not None:
                            print(f"==> Setting gradient of {n}.weight_v to None")
                            m.weight_v.grad = None
                    else:
                        print(f"==> No gradient to {n}.weight")
                        m.weight.requires_grad = False
                        if m.weight.grad is not None:
                            print(f"==> Setting gradient of {n}.weight to None")
                            m.weight.grad = None
                    
                    if not cfg.trainable_bias:
                        if hasattr(m, "bias") and m.bias is not None:
                            print(f"==> No gradient to {n}.bias")
                            m.bias.requires_grad = False

                            if m.bias.grad is not None:
                                print(f"==> Setting gradient of {n}.bias to None")
                                m.bias.grad = None
        
        if len(cfg.layerwise_prune_rate) > 0:
            assert len(cfg.layerwise_prune_rate) == 12
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "set_prune_rate"):
                    matched_layers = re.findall('layers.(\d+)', n)

                    if len(matched_layers) == 0: # do not apply masks on other layers
                        prune_rate = 1.0
                    else:
                        assert len(matched_layers) == 1

                        prune_rate = cfg.layerwise_prune_rate[int(matched_layers[0])]
                    
                    m.set_prune_rate(prune_rate)
                    print(f"==> Setting prune rate of {n} to {prune_rate}")
        
        elif len(cfg.fc_attn_prune_rate) > 0:
            assert len(cfg.fc_attn_prune_rate) == 2
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "set_prune_rate"):
                    if 'k_proj' in n or 'v_proj' in n or 'q_proj' in n or 'out_proj' in n:
                        prune_rate = cfg.fc_attn_prune_rate[1]

                    elif 'fc1' in n or 'fc2' in n:
                        prune_rate = cfg.fc_attn_prune_rate[0]

                    else:
                        print(f'There are other modules:{n}, in addition to fc/self-attn. Plz do not specify --fc_attn_prune_rate')
                        sys.exit()

                    m.set_prune_rate(prune_rate)
                    print(f"==> Setting prune rate of {n} to {prune_rate}")

        else:
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "set_prune_rate"):
                    if cfg.fix_attn:
                        if 'k_proj' in n or 'v_proj' in n or 'q_proj' in n or 'out_proj' in n:
                            prune_rate = 1.0
                            m.use_subset = False
                        else:
                            prune_rate = cfg.prune_rate

                    elif cfg.fix_fc:
                        if 'fc1' in n or 'fc2' in n:
                            prune_rate = 1.0
                            m.use_subset = False
                        else:
                            prune_rate = cfg.prune_rate

                    else:
                        prune_rate = cfg.prune_rate

                    m.set_prune_rate(prune_rate)
                    print(f"==> Setting prune rate of {n} to {prune_rate}")
        

        if cfg.ste_sigmoid:
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "enable_ste_sigmoid"):
                    m.enable_ste_sigmoid()

                    print(f"==> Enable Sigmoid for STE in {n}.")

        if cfg.init_score != 'kaiming_uniform':
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "init_score"):
                    if cfg.init_score == 'weight_magnitude_with_scale':
                        m.init_score(init_method=cfg.init_score, scale=cfg.init_score_scale)
                    else:
                        m.init_score(init_method=cfg.init_score)

                    print(f"==> Setting init score of {n} with:", cfg.init_score)

        
        if 'fix_mask' in cfg and cfg.fix_mask:
            for n, m in w2v_encoder.named_modules():
                if hasattr(m, "disable_score_grad"):
                    m.disable_score_grad()
                    print(f'Disable the gradient in the score of {n}')

        elif 'fix_mask_before' in cfg and cfg.fix_mask_before > 0:
            for n, m in w2v_encoder.named_modules():
                layer_id = re.findall('\.(\d+)\.', n)
                if len(layer_id) > 0 and int(layer_id[0]) < cfg.fix_mask_before and hasattr(m, "disable_score_grad"):
                    m.disable_score_grad()
                    print(f'Disable the gradient in the score of {n}')


        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x



class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        
        # assert cfg.normalize == w2v_args.task.normalize, (
        #     "Fine-tuning works best when data normalization is the same. "
        #     "Please check that --normalize is set or unset for both pre-training and here"
        # )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        
        w2v_args.model['_name'] = w2v_args.model['_name'] + '_st'  # redirect to new model definition
        # w2v_args.model['_name'] = 'wav2vec2_st'  # redirect to new model definition
        
        with open_dict(w2v_args):
            w2v_args.model.no_prune_extractor = cfg.no_prune_extractor
            w2v_args.model.no_prune_pos = cfg.no_prune_pos
            w2v_args.model.no_prune_post_extract_proj = cfg.no_prune_post_extract_proj

            w2v_args.model.linear_st_op = cfg.linear_st_op
                
        model = task.build_model(w2v_args.model)

        model.remove_pretraining_modules()
        
        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            if cfg.no_prune_proj:
                self.proj = Linear(d, targ_d)
            else:
                self.proj = Linear_ST(d, targ_d) # Linear(d, targ_d)


    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)

        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
                
            missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)

            # print('Missing kyes:', missing_keys)
            # print('Unexpected keys:', unexpected_keys)

            for missing_key in missing_keys:
                assert 'scores' in missing_key or 'thres' in missing_key, f'missing {missing_key}'
            
            assert len(unexpected_keys) == 0, 'unexpected keys: %s' % unexpected_keys


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

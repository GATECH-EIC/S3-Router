# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING

from typing import Any, Optional, Tuple
from fairseq.models.wav2vec.operation_st import Linear_ST, Linear_ST_InStr, Linear_ST_OutStr, Conv1d_ST

from omegaconf import II, MISSING, open_dict
import sys
import re

@dataclass
class HubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to hubert model"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
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
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside hubert model"
        },
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
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class HubertCtcConfig(HubertAsrConfig):
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


@register_model("hubert_ctc_st", dataclass=HubertCtcConfig)
class HubertCtc(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder_ST(cfg, task.target_dictionary)

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
        
        # for n, m in w2v_encoder.named_modules():
        #     if hasattr(m, "scores") and m.weight is not None:
        #         print(f"{n} scores.requires_grad", m.scores.requires_grad)

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

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@dataclass
class HubertSeq2SeqConfig(HubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )


class HubertEncoder_ST(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
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
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])

        ## Setup learnable masks 
        w2v_args.model['_name'] = 'hubert_st'  # redirect to new model definition
        with open_dict(w2v_args):
            w2v_args.model.no_prune_extractor = cfg.no_prune_extractor
            w2v_args.model.no_prune_pos = cfg.no_prune_pos
            w2v_args.model.no_prune_post_extract_proj = cfg.no_prune_post_extract_proj

            w2v_args.model.linear_st_op = cfg.linear_st_op

        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            if cfg.no_prune_proj:
                self.proj = Linear(d, len(tgt_dict))
            else:
                self.proj = Linear_ST(d, targ_d) # self.proj = Linear(d, len(tgt_dict))

        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
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

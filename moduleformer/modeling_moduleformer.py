""" PyTorch ModuleFormer model."""

from typing import Optional, Tuple, Union
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn import functional as F

from transformers.activations import get_activation
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_moduleformer import ModuleFormerConfig
from .utils.moe import MoE


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "moduleformer-small"
_CONFIG_FOR_DOC = "ModuleFormerConfig"


# SPARSEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "moduleformer-small",
#     # See all ModuleFormer models at https://huggingface.co/models?filter=moduleformer
# ]


@torch.jit.script
def stickbreaking_att(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor, 
    cum_weight: torch.Tensor,
    att_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute stick-breaking attention weights.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        mask (torch.Tensor): Mask tensor.
        cum_weight (torch.Tensor): Cumulative weight tensor.
        att_mask (Optional[torch.FloatTensor]): Attention mask tensor (default: None).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and attention weights.
    """
    logits = torch.einsum('bikhd,bjhd->bkhij', q, k) / math.sqrt(k.size(-1))
    mask = (mask[None, None, None, :, :] == 0).expand_as(logits)
    logits = logits + att_mask if att_mask is not None else logits
    z = F.sigmoid(logits).masked_fill(mask, 0)
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0)
    re_cum_log_beta = torch.einsum('bnhij,jk->bnhik', log_beta, cum_weight)
    att = z * re_cum_log_beta.exp()
    y = torch.einsum('bkhij,bjhd->bikhd', att, v)
    return y, att


class ModuleFormerAttention(nn.Module):
    def __init__(self, config):
        """
        Initialize the ModuleFormerAttention module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        
        self.q_proj = MoE(
                input_size=config.n_embd, 
                head_size=config.att_hidden, 
                num_experts=config.n_att_experts, 
                top_k=config.k_att,
                acc_aux_loss=False, 
                bias=False,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
                gate_type=config.gate_type,
            )
        if config.att_hidden == config.n_embd and config.n_head == 1:
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
        else:
            self.k_proj = nn.Linear(config.n_embd, config.att_hidden)
            self.v_proj = nn.Linear(config.n_embd, config.att_hidden)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.context_length = config.history_length + config.block_size

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(self.context_length, self.context_length, dtype=torch.int8))
        )
        self.register_buffer(
            "cum_weight", 
            torch.tril(torch.ones(self.context_length, self.context_length), -1)
        )
        self.n_head = config.n_head
        self.top_k = config.k_att
        self.n_embd = config.n_embd
        self.att_hidden = config.att_hidden
        self.head_size = config.att_hidden // config.n_head

    def add_history(self, k, v, hidden, use_cache=False):
        """
        Add history to key and value tensors.

        Args:
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            hidden: Hidden state.
            use_cache (bool): Whether to use cached history.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Updated key, value, and history.
        """
        if hidden is None or not use_cache:
            new_k = k
            new_v = v
        else:
            k_history, v_history = hidden
            new_k = torch.cat([k_history, k], dim=1)
            new_v = torch.cat([v_history, v], dim=1)
        k_history = new_k.detach()
        v_history = new_v.detach()

        return new_k, new_v, (k_history, v_history)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        """
        Forward pass of the ModuleFormerAttention module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            head_mask (Optional[torch.FloatTensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[...]]]: Tuple containing outputs.
        """
        B, T, C = hidden_states.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values 
        q, aux_loss = self.q_proj.map(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        k, v, hidden = self.add_history(k, v, layer_past, use_cache)
        context_length = k.size(1)
        
        q = q.view(B, T, self.top_k, self.n_head, self.head_size) # (B, T, k, nh, hs)
        k = k.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)
        v = v.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)

        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.int8, device=q.device))[context_length - T:, :]
        cum_weight=torch.tril(torch.ones(context_length, context_length, device=q.device), -1).type_as(q)

        y, attn_weights = stickbreaking_att(q, k, v, mask=mask, cum_weight=cum_weight, att_mask=attention_mask)

        # output projection
        y = self.q_proj.reduce(y.reshape(B, T, self.top_k, self.att_hidden).type_as(hidden_states))

        y = y.view(B, T, C) # re-assemble all head outputs side by side

        outputs = (y, hidden, aux_loss)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class ModuleFormerBlock(nn.Module):
    def __init__(self, config):
        """
        Initialize the ModuleFormerBlock module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = ModuleFormerAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = MoE(
                input_size=config.n_embd, 
                head_size=config.ffd_hidden, 
                num_experts=config.n_mlp_experts, 
                top_k=config.k_mlp, 
                bias=False, 
                activation=get_activation(config.activation_function),
                acc_aux_loss=False,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
                gate_type=config.gate_type,
            )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def get_aux_loss_and_clear(self):
        """
        Get auxiliary loss and clear auxiliary loss accumulators in the attention and MLP layers.

        Returns:
            torch.Tensor: Auxiliary loss.
        """
        return self.attn.q_proj.get_aux_loss_and_clear() + self.mlpf.get_aux_loss_and_clear()


    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """
        Forward pass of the ModuleFormerBlock module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            head_mask (Optional[torch.FloatTensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
            Tuple containing outputs or optional attention weights.
        """
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        hidden = attn_outputs[1]
        att_aux_loss = attn_outputs[2]

        hidden_states = hidden_states + self.resid_dropout(attn_output)
        x_mlp, mlp_aux_loss = self.mlpf(self.ln_2(hidden_states))
        hidden_states = hidden_states + self.resid_dropout(x_mlp)

        aux_loss = att_aux_loss + mlp_aux_loss
        return (hidden_states, hidden, aux_loss) + attn_outputs[3:]


class ModuleFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ModuleFormerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModuleFormerBlock"]

    def __init__(self, *inputs, **kwargs):
        """
        Initialize the ModuleFormerPreTrainedModel.

        Args:
            *inputs: Variable length input arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*inputs, **kwargs)

        self.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                self._set_gradient_checkpointing(
                    module, True, gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                self._set_gradient_checkpointing(
                    module, False
                )

    def _set_gradient_checkpointing(
        self,
        module,
        value=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    ):
        """
        Set gradient checkpointing for the ModuleFormerModel.

        Args:
            module: The module for which gradient checkpointing is set.
            value (bool): Whether to enable gradient checkpointing.
        """
        if isinstance(module, ModuleFormerModel):
            module.gradient_checkpointing = value
            module.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


SPARSEGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ModuleFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SPARSEGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoProcenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ModuleFormer Model transformer outputting raw hidden-states without any specific head on top.",
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerModel(ModuleFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ModuleFormerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(SPARSEGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        self.aux_loss = 0
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    **self.gradient_checkpointing_kwargs,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            self.aux_loss = self.aux_loss + outputs[2]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The ModuleFormer Model transformer with a language modeling head on top.
    """,
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerForCausalLM(ModuleFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.causal_mask"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ModuleFormerModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.aux_loss_weight = config.aux_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(SPARSEGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

            if self.aux_loss_weight > 0:
                loss = loss + self.transformer.aux_loss * self.aux_loss_weight

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

@add_start_docstrings(
    """
    The ModuleFormer Model with a sequence classification head on top (linear layer).

    [`ModuleFormerForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerForSequenceClassification(ModuleFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        # r"h\.\d+\.attn\.masked_bias", 
        r"lm_head.weight"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ModuleFormerModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SPARSEGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
""" ModuleFormer model configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging


logger = logging.get_logger(__name__)


# SPARSEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "moduleformer-small": "https://huggingface.co/moduleformer-small/resolve/main/config.json",
# }



class ModuleFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModuleFormerModel`]. It is used to instantiate a
    ModuleFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ModuleFormer
    [moduleformer-small](https://huggingface.co/moduleformer-small) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50400):
            Vocabulary size of the ModuleFormer model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModuleFormerModel`].
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            Number of dimensions in the embedding that Rotary Position Embedding is applied to.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import ModuleFormerConfig, ModuleFormerModel

    >>> # Initializing a ModuleFormer 6B configuration
    >>> configuration = ModuleFormerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ModuleFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "moduleformer"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50295,
        block_size=512,
        history_length=512,
        n_embd=1024,
        n_layer=24,
        n_head=8,
        att_hidden = 512,
        ffd_hidden=2048,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        moe_pdrop=0.0,
        sample_topk = 0,
        gating_size = 256,
        n_att_experts = 32,
        k_att = 2,
        n_mlp_experts = 32,
        k_mlp = 2,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        aux_loss_type = 'mi',
        aux_loss_weight=0,
        gate_type = "mlp",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.history_length = history_length
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.att_hidden = att_hidden
        self.ffd_hidden = ffd_hidden
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.moe_pdrop = moe_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sample_topk = sample_topk
        self.gating_size = gating_size
        self.n_att_experts = n_att_experts
        self.k_att = k_att
        self.n_mlp_experts = n_mlp_experts
        self.k_mlp = k_mlp
        self.aux_loss_type = aux_loss_type
        self.aux_loss_weight = aux_loss_weight
        self.gate_type = gate_type
        self.n_ctx = history_length * n_layer

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


class ModuleFormerOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        """
        Initialize the ModuleFormerOnnxConfig.

        Args:
            config (PretrainedConfig): Pretrained model configuration.
            task (str): Task description.
            patching_specs (List[PatchingSpec]): List of patching specifications.
            use_past (bool): Whether to use past tokens in the configuration.
        """
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Define the input mappings.

        Returns:
            Mapping[str, Mapping[int, str]]: Input mappings.
        """
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        """
        Get the number of layers.

        Returns:
            int: Number of layers.
        """
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        """
        Get the number of attention heads.

        Returns:
            int: Number of attention heads.
        """
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        """
        Generate dummy inputs for testing.

        Args:
            tokenizer (PreTrainedTokenizer): Pretrained tokenizer.
            batch_size (int): Batch size.
            seq_length (int): Sequence length.
            is_pair (bool): Whether the input is a pair.
            framework (Optional[TensorType]): Tensor framework.

        Returns:
            Mapping[str, Any]: Dummy inputs.
        """
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        """
        Get the default ONNX opset version.

        Returns:
            int: Default ONNX opset version.
        """
        return 13

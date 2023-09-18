import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_experts import ParallelExperts
from .gate import top_k_gating, compute_gating


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    

    Args:
        input_size: integer - size of the input
        head_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        acc_aux_loss: a boolean - whether to accumulate auxiliary loss
        hidden_size: an integer - hidden size of the experts
        gating_dropout: a float - dropout rate for gating network
        sample_topk: an integer - how many experts to sample during training
        gating_size: an integer - size of the gating network
        aux_loss: a string - type of auxiliary loss ('mi' or 'sparse')
        gate_type: a string - type of gating mechanism ('mlp' or 'topk')
    """

    def __init__(
        self, 
        input_size, 
        head_size, 
        num_experts, 
        top_k,
        bias=False, 
        activation=None, 
        acc_aux_loss=False,
        hidden_size=None,
        gating_dropout=0.0,
        sample_topk=0,
        gating_size=256,
        aux_loss='mi',
        gate_type='mlp',
        ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        if hidden_size is None:
            hidden_size = head_size
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.gate = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            acc_aux_loss=acc_aux_loss, 
            dropout=gating_dropout,
            sample_topk=sample_topk,
            hidden_size=gating_size,
            aux_loss=aux_loss,
            gate_type=gate_type,
            )

    def extra_repr(self):
        return 'k={}'.format(
            self.top_k)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, moe_inp, skip_mask=None):
        """
        Compute gating for the mixture of experts.

        Args:
            moe_inp (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.

        Returns:
            float: Gating loss.
        """

        top_k_indices, top_k_gates, probs = self.gate(moe_inp, skip_mask=skip_mask)
        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts =\
            compute_gating(self.top_k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size.tolist()
        return self.gate.loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
            skip_mask = skip_mask.flatten()[:, None]
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x, skip_mask)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, self.input_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        """
        
        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.compute_gate(x, skip_mask)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.top_k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """
        
        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y
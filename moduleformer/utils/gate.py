import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

# @torch.jit.script
def log_gmm_posterior(z, expert_centroids):
     """
    Compute the log posterior probabilities of data points z belonging to Gaussian mixture components defined by centroids.

    Args:
        z (torch.Tensor): Data points (batch_size x feature_dim).
        expert_centroids (torch.Tensor): Centroids of Gaussian mixture components (num_experts x feature_dim).

    Returns:
        torch.Tensor: Log posterior probabilities for each data point (batch_size x num_experts).
    """
     return (
        torch.matmul(z, expert_centroids.t())
        # - 0.5 * (
        #     torch.einsum('ni,ni->n', z, z)[:, None] +
        #     torch.einsum('ni,ni->n', expert_centroids, expert_centroids)[None, :]
        # )
     )


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        probs (torch.Tensor): Probability values for each expert (batch_size x num_experts).
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class top_k_gating(nn.Module):
    def __init__(
        self,
        input_size, 
        num_experts, 
        top_k,
        acc_aux_loss=False, 
        dropout=0.1,
        hidden_size=256,
        sample_topk=0,
        aux_loss='mi',
        gate_type='mlp',
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k
        assert sample_topk <= top_k
        self.sample_topk = sample_topk

        self.acc_aux_loss = acc_aux_loss
        self.aux_loss = aux_loss
        self.init_aux_statistics()

        self.gate_type = gate_type
        if gate_type == 'mlp':
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts, bias=False)
            )
        elif gate_type == 'linear':
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, num_experts, bias=False)
            )
        elif gate_type == 'gmm':
            self.w_gate = nn.Linear(input_size, hidden_size, bias=False)
            self.expert_centroids = nn.Parameter(torch.empty(num_experts, hidden_size))
            nn.init.normal_(self.expert_centroids)
            self.temperature = nn.Parameter(torch.zeros(1))
        else:
            print(gate_type)
            raise NotImplementedError

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return 'k={}, num_experts={}, aux_loss={}'.format(
            self.top_k, self.num_experts, self.aux_loss)

    def init_aux_statistics(self):
        """
        Initialize auxiliary statistics based on the chosen auxiliary loss type.
        """
        if self.aux_loss == 'mi':
            self.p_e = 0.
            self.neg_H_e_given_x = 0.
            self.count_layers = 0
        else:
            self.acc_probs = 0.
            self.acc_freq = 0.
            self.acc_lsesq = 0.
            self.acc_count = 0

    def update_aux_statistics(self, probs, logits, gates, skip_mask=None):
        """
        Update auxiliary statistics based on the current batch.

        Args:
            probs (torch.Tensor): Probability values for each expert.
            logits (torch.Tensor): Logits values for each expert.
            gates (torch.Tensor): Gating values for each expert.
            skip_mask (torch.Tensor): Skip mask tensor.

        """
        if self.aux_loss == 'mi':
            log_prob = torch.log_softmax(logits, dim=-1)
            self.p_e = self.p_e + probs.mean(0)
            self.neg_H_e_given_x = self.neg_H_e_given_x + (probs * log_prob).sum() / probs.size(0)
            self.count_layers += 1
        else:
            self.acc_count = self.acc_count + logits.size(0)
            self.acc_probs = self.acc_probs + probs.sum(0)
            self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
            lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
            self.acc_lsesq = self.acc_lsesq + lsesq.sum()

    def get_aux_loss_and_clear(self, eps=1e-8):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        if self.aux_loss == 'mi':
            denominator = self.count_layers 
            p_e = self.p_e / denominator
            H_e = -(p_e * (p_e + eps).log()).sum()
            neg_H_e_given_x = self.neg_H_e_given_x / denominator
            miloss = -(neg_H_e_given_x + H_e)
            loss = miloss
        else:
            switchloss =  self.num_experts * (
                F.normalize(self.acc_probs, p=1, dim=0) *
                F.normalize(self.acc_freq, p=1, dim=0)
            ).sum()
            zloss = self.acc_lsesq / self.acc_count
            loss = switchloss + 0.1 * zloss

        self.init_aux_statistics()
        return loss

    def forward(self, x, skip_mask=None):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        if self.gate_type in ['linear', 'mlp']:
            logits = self.w_gate(x)
        elif self.gate_type == 'gmm':
            z = self.w_gate(x)
            logits = log_gmm_posterior(F.normalize(z, p=2, dim=-1), F.normalize(self.expert_centroids, p=2, dim=-1)) * self.temperature.exp()

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, (skip_mask == 0), 0)
            logits = torch.masked_fill(logits, (skip_mask == 0), 0)

        if self.training and (self.sample_topk > 0):
            _, top_km1_indices = probs.topk(self.top_k - self.sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, self.sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.top_k, dim=1)

        # if self.top_k > 1:
        #     top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)
        
        # gate = torch.zeros_like(top_k_gates)
        # gate[:, 0] = 1
        # top_k_gates = (gate - top_k_gates).detach() + top_k_gates

        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.update_aux_statistics(probs, logits, gates, skip_mask)
        if not self.acc_aux_loss:
            self.loss = self.get_aux_loss_and_clear()
        else:
            self.loss = 0

        return top_k_indices, top_k_gates, probs

        # batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
        #     compute_gating(self.top_k, probs, top_k_gates, top_k_indices)

        # return batch_gates, batch_index, expert_size.tolist(), gates, index_sorted_experts
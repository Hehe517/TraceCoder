from typing import Optional, Callable

from opt_einsum import contract
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.init as init
from typing import Union, Tuple


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

class MultiSynonymsAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(MultiSynonymsAttention, self).__init__()
        self.d_k = int(d_model / n_heads)
        self.d_v = self.d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_k * n_heads, bias=False)

        self._init_weights(mean=0.0, std=0.001)

    def _init_weights(self, mean: float = 0.0, std: float = 0.001) -> None:
        """
        Initialise the weights of the Linear layers in MultiSynonymsAttention.

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.001.
        """

        torch.nn.init.normal_(self.W_Q.weight, mean, std)
        torch.nn.init.normal_(self.W_K.weight, mean, std)
        torch.nn.init.normal_(self.W_V.weight, mean, std)


    def forward(self, Q, H, ql):
        device = Q.device
        Q, H, ql = Q.to(device), H.to(device), ql.to(device)
        self.W_Q = self.W_Q.to(device)
        self.W_K = self.W_K.to(device)
        self.W_V = self.W_V.to(device)

        n_classes = Q.size(0)
        batch_size = H.size(0)

        q_s = self.W_Q(Q).view(n_classes, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(H).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        Wql = self.W_V(ql)

        H = H.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores, context, attn = SynonymsScaledDotProductAttention(self.d_k).to(device)(q_s, k_s, H)

        context = context.view(batch_size, n_classes, self.n_heads * self.d_v)

        output = contract("bch,ch->bc", context, Wql)

        return output


class SynonymsScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SynonymsScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, WQ, WK, K):
        device = WQ.device
        WK = nn.Tanh()(WK.to(device))
        K = nn.Tanh()(K.to(device))

        scores = contract('bzth,czsh->bczst', WK, WQ)
        scores = scores.to(device)
        attn = nn.Softmax(dim=-1)(scores)
        context = contract('bzth,bczst->bczsh', K, attn)

        context = nn.AvgPool3d((1, context.shape[3], 1)).to(device)(context).squeeze(-2)

        return scores, context, attn


class LabelCrossAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
            self,
            x: torch.Tensor,
            attention_masks: Optional[torch.Tensor] = None,
            output_attention: bool = False,
            attn_grad_hook_fn: Optional[Callable] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:


        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        att_weights = Q.matmul(K.transpose(1, 2))

        if attention_masks is not None:
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=1
            )
            attention_masks = attention_masks.to(torch.bool)

            attention_masks = attention_masks.unsqueeze(1).repeat(
                1, self.num_classes, 1
            )
            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:


        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )
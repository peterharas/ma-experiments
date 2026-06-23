from typing import Any, Dict, Optional, Tuple

import torch
from torch.compiler import F
import torch.nn as nn

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    This module applies a linear transformation to the input and then splits
    the result into two parts. It applies a sigmoid activation to one part
    and performs element-wise multiplication with the other part.

    Args:
        input_dimension (int): The dimensionality of the input tensor.
        output_dimension (int, optional): The dimensionality of the output tensor.
            If not provided, it defaults to the input_dimension.
        dropout (float, optional): Dropout rate to apply to the input tensor.
            If None, no dropout is applied. Defaults to None.

    Attributes:
        dropout (nn.Dropout or None): Dropout layer if dropout rate is provided.
        dense (nn.Linear): Linear layer for transforming the input.

    Methods:
        forward(upgamma):
            Applies the linear transformation and GLU operation to the input tensor.

            Args:
                upgamma (torch.Tensor): Input tensor of shape
                    (batch_size, ..., input_dimension).

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, ..., output_dimension).
    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super(GatedLinearUnit, self).__init__()

        # Create a valid dropout layer or set to None
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        # Linear projection of the eta
        output_dim = output_dimension or input_dimension
        self.dense = nn.Linear(input_dimension, output_dim * 2)

    def forward(self, upgamma: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            upgamma = self.dropout(upgamma)

        upgamma = self.dense(upgamma)
        return F.glu(
            upgamma, dim=-1
        )  # reduces the dimensionality to the original dimensionality
    

class AddNorm(nn.Module):
    """
    Add & Normalize module.

    This module implements a combination of addition and layer normalization.
    It's designed to add a skip connection to the input and then apply layer
    normalization to the result.

    Args:
        dimension (int): The dimensionality of the input tensor.

    Attributes:
        norm (nn.LayerNorm): Layer normalization module.

    Methods:
        forward(x, skip):
            Applies the skip connection and layer normalization to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, ..., dimension).
                skip (torch.Tensor): Skip connection tensor of shape
                    (batch_size, ..., dimension).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, ..., dimension).


    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(self, dimension: int):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Any:
        # Assumes x.shape == skip.shape
        return self.norm(x + skip)
    

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) module.

    This module implements a Gated Residual Network, which is a type of neural
    network layer that combines residual connections with gating mechanisms.
    It's designed to process input data through a series of linear transformations
    and non-linear activations, with the ability to incorporate context information.

    Args:
        input_size (int): The dimensionality of the input tensor.
        hidden_size (int): The dimensionality of the hidden layer.
        output_size (int): The dimensionality of the output tensor.
        dropout (float): Dropout rate to apply to the input tensor.
        context_size (int): The dimensionality of the context tensor.

    Attributes:
        input_proj (nn.Linear): Linear layer for projecting the input to the hidden size.
        elu (nn.ELU): ELU activation function.
        context_proj (nn.Linear or None): Linear layer for projecting the context to the hidden size.
        fully_connected2 (nn.Linear): Linear layer for further processing the hidden representation.
        GLU (GatedLinearUnit): Gated Linear Unit for gating mechanism.
        add_norm (AddNorm): Add & Normalize layer for residual connections.

    Methods:
        forward(a, context=None):
            Applies the GRN to the input tensor.

            Args:
                a (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
                context (torch.Tensor, optional): Context tensor of shape (batch_size, context_size).
                    Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_size).

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        context_size: int,
    ):
        super(GatedResidualNetwork, self).__init__()

        # input projection layer
        # [batch size, sequence length, input size] -> [batch size, sequence length, output size]
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        # Context projection layer
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        # eta 1
        self.fully_connected2 = nn.Linear(hidden_size, hidden_size)
        self.GLU = GatedLinearUnit(hidden_size, output_size, dropout=dropout)
        self.add_norm = AddNorm(hidden_size)

    def forward(
        self, a: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # a: [batch size, sequence length, input size]
        a = self.input_proj(a)

        residual = a

        if context is not None:
            # Equation (4)
            # project context to hidden dim
            context = self.context_proj(context)
            # expand over sequence length
            context = context.unsqueeze(1).expand(  # type: ignore[union-attr]
                -1, a.shape[1], -1
            )  # expand the context
            a = a + context
        a = self.elu(a)
        a = self.fully_connected2(a)

        a = self.GLU(a)
        a = self.add_norm(a, residual)

        return a  # [batch size, sequence length, output size]


class GatedAddNorm(nn.Module):
    """
    Gated Add & Norm module.

    This module combines a Gated Linear Unit (GLU) with an Add & Norm operation.
    It's designed to process input data through a GLU and then apply a residual
    connection with layer normalization, similar to the AddNorm module but with
    an added gating mechanism.

    Args:
        input_dimension (int): The dimensionality of the input tensor.
        hidden_dimensions (int): The dimensionality of the hidden layer.
        output_dimension (int): The dimensionality of the output tensor.
        skip_dimension (int, optional): The dimensionality of the skip connection tensor.
            If not provided, it defaults to the output_dimension.
        dropout (float, optional): Dropout rate to apply to the input tensor.
            Defaults to 0.1.

    Attributes:
        skip_dimension (int): The dimensionality of the skip connection tensor.
        skip_layer_proj (nn.Linear or None): Linear layer for projecting the skip
            connection tensor to the output dimension if necessary.
        GLU (GatedLinearUnit): Gated Linear Unit for gating mechanism.
        add_norm (AddNorm): Add & Normalize layer for residual connections.

    Methods:
        forward(x, skip):
            Applies the skip connection projection (if necessary), GLU, and
            Add & Norm operation to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, ..., input_dimension).
                skip (torch.Tensor): Skip connection tensor of shape
                    (batch_size, ..., skip_dimension).

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, ..., output_dimension).

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: int,
        output_dimension: int,
        skip_dimension: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super(GatedAddNorm, self).__init__()

        self.skip_dimension = skip_dimension

        if skip_dimension is not None and skip_dimension != output_dimension:
            self.skip_layer_proj = nn.Linear(skip_dimension, output_dimension)  # type: ignore
        else:
            self.skip_layer_proj = None  # type: ignore[assignment]

        self.GLU = GatedLinearUnit(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            dropout=dropout,
        )
        self.add_norm = AddNorm(hidden_dimensions)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Any:
        if self.skip_layer_proj is not None:
            # skip is of a different dimensionality, project first
            skip = self.skip_layer_proj(skip)

        return self.add_norm(self.GLU(x), skip)
    

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) module.

    This module implements a Variable Selection Network, which is a type of
    neural network layer designed to select relevant input variables from a
    set of multiple input features. It uses Gated Residual Networks (GRNs)
    to process each input feature and then combines them using a softmax-based
    attention mechanism.

    Args:
        inputs (Dict[str, int]): A dictionary where keys are input names and
            values are their respective dimensions.
        hidden_dimensions (int): The dimensionality of the hidden layers in the GRNs.
        dropout (float): Dropout rate to apply to the input tensor.
        context_size (int): The dimensionality of the context tensor.

    Attributes:
        inputs_length (int): The number of input features.
        input_grns (nn.ModuleDict): A dictionary of GRNs, one for each input feature.
        input_size_total (int): The total dimensionality of all input features.
        grn_input (GatedResidualNetwork): GRN for processing the concatenated input features.
        softmax (nn.Softmax): Softmax layer for computing attention weights.

    Methods:
        forward(x, context=None):
            Applies the VSN to the input features.

            Args:
                x (Dict[str, torch.Tensor]): A dictionary of input tensors, where keys
                    are input names and values are tensors of shape
                    (batch_size, sequence_length, input_dimension).
                context (torch.Tensor, optional): Context tensor of shape
                    (batch_size, context_size). Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, sequence_length, hidden_dimensions).

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self,
        inputs: Dict[str, int],
        hidden_dimensions: int,
        dropout: float,
        context_size: int,
    ) -> None:
        super(VariableSelectionNetwork, self).__init__()

        self.inputs_length = len(inputs)

        # Loop over all inputs, create a GRN per input
        self.input_grns = nn.ModuleDict()
        self.input_size_total = 0
        for key, input_dim in inputs.items():
            self.input_size_total += input_dim
            # Equation (7)
            self.input_grns[key] = GatedResidualNetwork(
                input_size=input_dim,
                hidden_size=hidden_dimensions,
                output_size=hidden_dimensions,
                dropout=dropout,
                context_size=context_size,
            )

            # The GRN for the softmax, Equation (6)
        self.grn_input = GatedResidualNetwork(
            input_size=self.input_size_total,
            hidden_size=hidden_dimensions,
            output_size=hidden_dimensions,
            dropout=dropout,
            context_size=context_size,
        )

        # Equation (6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None
    ) -> Any:
        # x: dictionary with tensors of different dimensionality
        # x[input]: Tensor => [batch size, sequence length, input feature dimension]
        # Context: Tensor => [batch size, context size]
        # Context size is linearly projected to hidden dimensionality, and expanded to the sequence length

        # Handle empty input case
        if len(self.input_grns) == 0:
            # Return a zero tensor with the expected output shape
            # Determine batch size and sequence length from context or create dummy values
            if context is not None:
                batch_size = context.shape[0]
                seq_len = 1  # Default sequence length when no inputs
            else:
                batch_size = 1
                seq_len = 1
            device = context.device if context is not None else torch.device('cpu')
            return torch.zeros(batch_size, seq_len, self.hidden_dimensions, device=device)

        transformed_values_output = []
        values_output = []
        for key, grn in self.input_grns.items():
            # Equation (7)
            transformed_values_output.append(grn(x[key], context))
            values_output.append(x[key])

        # Create the XI matrix
        transformed_inputs = torch.stack(transformed_values_output, dim=-2)
        XI_embedding = torch.cat(values_output, dim=-1)

        # Equation (6)
        v_chi = self.softmax(self.grn_input(XI_embedding, context)).unsqueeze(-2)

        # Equation (8)
        output = transformed_inputs * v_chi
        # output => [batch size, sequence length, inputs, hidden]
        result = output.sum(dim=-2)  # sum over the input dimensions
        # result => [batch size, sequence length, hidden]
        return result
    

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    This module implements the scaled dot-product attention mechanism,
    which computes the attention weights by taking the dot product of
    the query and key matrices, scaling the result, and then applying
    a softmax function.
    Args:
        dropout (float): Dropout probability applied to the attention weights.
    Attributes:
        dropout (nn.Dropout): Dropout layer.
        softmax (nn.Softmax): Softmax layer for normalizing attention weights.
    Methods:
        forward(query, key, value, mask):
            Computes the forward pass of the scaled dot-product attention.
            Args:
                query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embedding_dim).
                key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embedding_dim).
                value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embedding_dim).
                mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len),
                                     where masked positions are indicated by a value of 1.
            Returns:
                output (torch.Tensor): The output tensor of shape (batch_size, seq_len, embedding_dim).
                attention (torch.Tensor): The attention weights of shape (batch_size, seq_len, seq_len).
        mask_attention(attention, mask):
            Applies a mask to the attention weights.
            Args:
                attention (torch.Tensor): Attention weights tensor.
                mask (torch.Tensor): Mask tensor.
            Returns:
                torch.Tensor: Masked attention weights.

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(self, dropout: float) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # perform a softmax on the last dimenion (transformed sequence output)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query:
        # k (batch_size, seq_length, embedding_dim)
        # q (batch_size, seq_length, embedding_dim)
        # v (batch_size, seq_length, embedding_dim)

        # First calculate QK^T, we have (batch_size, seq_length, embedding_dim) as inputs
        # using torch.bmm (batch matrix multiplication)
        # which expects (batch_size, n, m) and (batch_size, m, p) to return (batch_size, n, p)
        # transform k from (batch_size, seq_length, embedding_dim) => (batch_size, embedding_dim, seq_length)
        # such that we can use bmm with output (batch_size, seq_length, seq_length)
        attention = torch.bmm(query, key.permute(0, 2, 1))

        # scale factor of equation (10)
        embedding_dim = key.size(-1)
        d_attention = torch.as_tensor(
            embedding_dim, dtype=attention.dtype, device=attention.device
        ).sqrt()
        # Scaled attention
        attention = attention / d_attention
        attention = (
            self.mask_attention(attention, mask) if mask is not None else attention
        )

        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # attention: (batch_size, seq_length, seq_length)
        # value: (batch_size, seq_length, embedding_dim)
        # output bmm: (batch_size, seq_length, embedding_dim)
        # output attention: (batch_size, seq_length, seq_length)
        return torch.bmm(attention, value), attention

    def mask_attention(
        self, attention: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Attention (batch_size, seq_length, seq_length)
        # mask (batch_size, seq_length, seq_length)
        mask = mask.to(attention.device)
        return attention.masked_fill(mask, -1e9)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention module.

    This module implements the interpretable multi-head attention mechanism,
    which is a variant of the standard multi-head attention that allows
    for the interpretation of the attention weights. It divides the input
    into multiple heads and computes attention independently for each head,
    then combines the results.

    Args:
        number_of_heads (int): The number of attention heads.
        input_dimension (int): The dimensionality of the input tensor.
        dropout (float): Dropout rate to apply to the attention weights.

    Attributes:
        number_of_heads (int): The number of attention heads.
        dropout (nn.Dropout): Dropout layer.
        value_projection (nn.Linear): Linear layer for projecting the value tensor.
        query_projections (nn.ModuleList): List of linear layers for projecting the query tensor for each head.
        key_projections (nn.ModuleList): List of linear layers for projecting the key tensor for each head.
        attention (ScaledDotProductAttention): Scaled dot-product attention module.
        final_projection (nn.Linear): Linear layer for projecting the concatenated attention outputs.

    Methods:
        forward(q, k, v, mask):
            Computes the forward pass of the interpretable multi-head attention.

            Args:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dimension).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dimension).
                v (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dimension).
                mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len),
                                     where masked positions are indicated by a value of 1.

            Returns:
                output (torch.Tensor): The output tensor of shape (batch_size, seq_len, input_dimension).
                attentions (torch.Tensor): The attention weights of shape (batch_size, num_heads, seq_len, seq_len).

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self, number_of_heads: int, input_dimension: int, dropout: float
    ) -> None:
        super(InterpretableMultiHeadAttention, self).__init__()
        # Creates the dimensions for the attentions,
        # we equally divide the input over the number of heads
        self.number_of_heads = number_of_heads
        dim_key = dim_query = dim_value = input_dimension // number_of_heads

        self.dropout = nn.Dropout(dropout)

        # Linear projection value
        self.value_projection = nn.Linear(input_dimension, dim_value)
        # Query projections for each head
        self.query_projections = nn.ModuleList(
            [nn.Linear(input_dimension, dim_query) for _ in range(number_of_heads)]
        )
        # Key projection for each head
        self.key_projections = nn.ModuleList(
            [nn.Linear(input_dimension, dim_key) for _ in range(number_of_heads)]
        )

        self.attention = ScaledDotProductAttention(dropout)
        self.final_projection = nn.Linear(dim_value, input_dimension, bias=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q = k = v = mask => [batch size, sequence length, hidden]
        heads_outputs = []
        attn_outputs = []

        v_proj = self.value_projection(v)
        # Generate a forward-pass for each head
        for head in range(self.number_of_heads):
            # Do a linear projection for all inputs
            q_head = self.query_projections[head](q)
            k_head = self.key_projections[head](k)

            # Feed the unique head projection, to the head attention mechanism
            # Note that we have a single attention mechanism for all heads
            # Equation (12)
            output, attention = self.attention(q_head, k_head, v_proj, mask)
            output_dropout = self.dropout(output)

            heads_outputs.append(output_dropout)
            attn_outputs.append(attention)

        # stack the output of all heads along the embedding dimension
        heads = torch.stack(heads_outputs, dim=2)
        # stack the attentions of along the sequence dimension
        attentions_stacked = torch.stack(attn_outputs, dim=2)

        # In accordance with equations (14), (15), and (16)
        # take the mean head embedding
        mean_head = heads.mean(dim=2)
        output = self.final_projection(mean_head)
        output = self.dropout(output)

        # [batch size, sequence length, hidden]
        # [batch size, heads, sequence lenght, sequence length, sequence length]
        return output, attentions_stacked.permute(0, 2, 1, 3)
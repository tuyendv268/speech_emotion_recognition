from torch.nn.utils import weight_norm
import torch.nn.functional as F 
from typing import Optional
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.init as init

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, slf_attn_masks: Tensor=None) -> Tensor:
        if slf_attn_masks is not None:
            return (self.module(inputs, slf_attn_masks) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class SpatialDropout1D(nn.Module):
    def __init__(self, drop_rate):
        super(SpatialDropout1D, self).__init__()
        
        self.dropout = nn.Dropout2d(drop_rate)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        inputs = self.dropout(inputs.unsqueeze(2)).squeeze(2)
        inputs = inputs.permute(0, 2, 1)
        
        return inputs

class Conv(nn.Module):
    def __init__(self, channels=[80, 256], kernels=[3, 3], dropout=0.2):
        super(Conv, self).__init__()
        convs = []
        
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            conv = [
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernels[i],
                    padding="same",
                    stride=1),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            ]            
            convs += conv
        
        self.convs = nn.ModuleList(convs)
        self.dropout = SpatialDropout1D(dropout)
    
    def forward(self, inputs):
        for conv in self.convs:
            inputs = conv(inputs)
        inputs = self.dropout(inputs)
        inputs = inputs.permute(0, 2, 1)
        return inputs

class AdditiveAttention(nn.Module):
    def __init__(self, dropout,
                 query_vector_dim,
                 candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        candidate_weights = self.dropout(candidate_weights)
            
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target


class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, slf_attn_masks: Tensor=None) -> Tensor:
        if slf_attn_masks is not None:
            return (self.module(inputs, slf_attn_masks) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 256,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 7,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.ffw_1 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        
        self.multi_head_attn = ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            )
        
        self.conformer_conv = ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            )
        
        self.ffw_2 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, inputs: Tensor, masks: Tensor, slf_attn_masks: Tensor) -> Tensor:
        if masks is not None:
            inputs = inputs.masked_fill(masks.unsqueeze(-1), 0)
        inputs = self.ffw_1(inputs)
        
        inputs = self.multi_head_attn(inputs, slf_attn_masks)
        
        if masks is not None:
            inputs = inputs.masked_fill(masks.unsqueeze(-1), 0)
        inputs = self.conformer_conv(inputs)
        
        if masks is not None:
            inputs = inputs.masked_fill(masks.unsqueeze(-1), 0)
        inputs = self.layer_norm(inputs)
        return inputs

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 128,
            num_layers: int = 4,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            max_seq_len: int=1024,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 7,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        
        n_position = max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, encoder_dim).unsqueeze(0),
            requires_grad=False,
        )
        
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual) 
                for _ in range(num_layers)]
            )
        
    def forward(self, inputs, masks):
        batch_size, max_len = inputs.shape[0], inputs.shape[1]
        slf_attn_mask = masks.unsqueeze(1).expand(-1, max_len, -1)
        
        outputs = inputs + self.position_enc[:, :max_len, :] \
                .expand(batch_size, -1, -1)
        
        hidden_states = []
        for enc_layer in self.layers:
            outputs = enc_layer(
                inputs=outputs, masks=masks, slf_attn_masks=slf_attn_mask
            )
                        
            hidden_states.append(outputs)
        hidden_states = torch.stack(hidden_states, dim=1)
        last_hidden_state = outputs

        return last_hidden_state, hidden_states



class CNN_Conformer(nn.Module):
    def __init__(self, config, n_label) -> None:
        super(CNN_Conformer, self).__init__()
        self.cnn = Conv(
            channels=config["cnn_channels"], 
            kernels=config["cnn_kernels"], 
            dropout=config["cnn_dropout"])
        
        self.conformer = ConformerEncoder(
            encoder_dim=config["encoder_dim"],
            num_layers=config["num_layers"],
            num_attention_heads=config["num_attention_heads"],
            feed_forward_expansion_factor=config["feed_forward_expansion_factor"],
            conv_expansion_factor=config["conv_expansion_factor"],
            feed_forward_dropout_p=config["feed_forward_dropout_p"],
            attention_dropout_p=config["attention_dropout_p"],
            conv_dropout_p=config["conv_dropout_p"],
            conv_kernel_size=config["conv_kernel_size"],
            half_step_residual=config["half_step_residual"]
        )
        self.weighted_layers = nn.Parameter(torch.randn(1, config["num_layers"]))
        
        self.additive_attention = AdditiveAttention(
            dropout=config["addi_attn_dropout"], 
            query_vector_dim=config["encoder_dim"],
            candidate_vector_dim=config["encoder_dim"])
        
        self.cls_head = nn.Linear(config["encoder_dim"], n_label)
        
    def get_mask_from_lengths(self, lengths, max_len=None, device=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
            
        if device is not None:
            ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        else:
            ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask
    
    def forward(self, inputs, masks):
        inputs = self.cnn(inputs)
        last_hidden_state, hidden_states = self.conformer(inputs, ~masks)
        
        batch_size, num_layer, seq_length, embedding_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, num_layer, -1)
        outputs = torch.matmul(self.weighted_layers, hidden_states)
        outputs = outputs.view(batch_size, seq_length, embedding_dim)
        
        seq_embeddings = outputs.contiguous()
        # embedding = self.additive_attention(seq_embeddings)
        embedding = seq_embeddings.mean(dim=1)

        output = self.cls_head(embedding)
        return None, output
        
        
if __name__ == "__main__":
    import yaml
    config = yaml.load(open("/home/tuyendv/Desktop/speech_emotion_recognition/configs/models/cnn_conformer.yml", "r"), Loader=yaml.FullLoader)

    model = CNN_Conformer(config)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    inputs = torch.randn(8, 64, 80)
    masks = torch.ones(8, 64)
    
    output = model(inputs, masks)
    print(output[0].shape)
    print(output[1].shape)
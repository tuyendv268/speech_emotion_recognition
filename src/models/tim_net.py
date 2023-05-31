import torch.nn.functional as F
from torch import nn
import torch

class CasualConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(CasualConv1D, self).__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation
        )
        
        self.padding_left = (kernel_size-1)*dilation
        
    def forward(self, inputs):
        inputs =  F.pad(inputs.unsqueeze(2), pad=(self.padding_left,0,0,0), value=0).squeeze(2)
        outputs = self.conv1d(inputs)
        return outputs
   
class SpatialDropout1D(nn.Module):
    def __init__(self, drop_rate):
        super(SpatialDropout1D, self).__init__()
        
        self.dropout = nn.Dropout2d(drop_rate)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        inputs = self.dropout(inputs.unsqueeze(2)).squeeze(2)
        inputs = inputs.permute(0, 2, 1)
        
        return inputs
    
    
class Temporal_Aware_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 drop_rate=0.1, stride=1, padding=0, dialation=1) -> None:
        super(Temporal_Aware_Block, self).__init__()
        
        self.conv_1 = nn.Sequential(
            *[
                CasualConv1D(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, dilation=dialation,
                             stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                SpatialDropout1D(drop_rate=drop_rate)
            ]
        )
        
        self.conv_1 = nn.ModuleList(self.conv_1)
        
        self.conv_2 = nn.Sequential(
            *[
                CasualConv1D(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, dilation=dialation,
                             stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                SpatialDropout1D(drop_rate=drop_rate),
                nn.Sigmoid()
            ]
        )
        self.conv_2 = nn.ModuleList(self.conv_2)
        
    def forward(self, inputs):
        outputs = inputs
        for layer in self.conv_1:
            outputs = layer(outputs)
            
        for layer in self.conv_2:
            outputs = layer(outputs)
                    
        return inputs * outputs
        
        
class TimNet(nn.Module):
    def __init__(self, n_filters=39, kernel_size=2, n_stacks=1, dialations=8,
                 drop_rate=0.1, n_label=8) -> None:
        super(TimNet, self).__init__()
        
        self.forward_convs = nn.ModuleList(
            nn.Sequential(
                *[
                    Temporal_Aware_Block(
                        in_channels=n_filters, out_channels=n_filters,
                        kernel_size=kernel_size, drop_rate=drop_rate,
                        padding=0, dialation=i
                        )
                    for i in [2**i for i in range(dialations)]
                    ]
                )
            )
        self.forward_conv = CasualConv1D(
            in_channels=n_filters, out_channels=n_filters, 
            kernel_size=1, dilation=1, padding=0
            )

        self.backward_convs = nn.ModuleList(
            nn.Sequential(
                *[
                    Temporal_Aware_Block(
                        in_channels=n_filters, out_channels=n_filters,
                        kernel_size=kernel_size, drop_rate=drop_rate,
                        padding=0, dialation=i)
                    
                    for i in [2**i for i in range(dialations)]
                    ]
                )
            )
        
        self.backward_conv = CasualConv1D(
            in_channels=n_filters, out_channels=n_filters, 
            kernel_size=1, dilation=1, padding=0
            )
        
        self.weight_layer = nn.Parameter(torch.randn(dialations, 1))
        self.cls_head = nn.Linear(n_filters, n_label)
        
    def forward(self, inputs, masks=None):
        forward_inputs = inputs
        backward_input = torch.flip(inputs, dims=(-1,))
        
        forward_tensor = self.forward_conv(forward_inputs)
        backward_tensor = self.backward_conv(backward_input)
        
        final_skip_connection = []
        
        skip_out_forward = forward_tensor
        skip_out_backward = backward_tensor
        
        for forward, backward in zip(self.forward_convs, self.backward_convs):
            skip_out_forward = forward(skip_out_forward)
            skip_out_backward = backward(skip_out_backward)
            
            tmp = skip_out_forward + skip_out_backward
            tmp = torch.mean(tmp, dim=2)
            tmp = torch.unsqueeze(tmp, dim=2)
            
            final_skip_connection.append(tmp)
        
        output_2 = final_skip_connection[0]
        for i, output in enumerate(final_skip_connection):
            if i == 0:
                continue
            
            output_2 = torch.cat([output_2, output], dim=2)
        
        output = torch.matmul(output_2, self.weight_layer).squeeze(-1)
        output = self.cls_head(output)
        return None, output
    
if __name__ == "__main__":
    model = TimNet()

    inputs = torch.randn(32, 39, 51)
    outputs = model(inputs)
    print(outputs[1].shape)
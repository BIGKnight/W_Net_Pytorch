import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as functional
vgg16bn_frop_fc = list(model.vgg16_bn(pretrained=False).children())[0]

class ConnectLayer(nn.Module):
    def __init__(self):
        super(ConnectLayer, self).__init__()
        
    def forward(self, decoded_input, encoded_input):
        decoded_input = functional.interpolate(decoded_input, scale_factor=[2, 2, 1], mode="nearest")
        return torch.cat([decoded_input, encoded_input], dim=1)

class BasicConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 pad, 
                 if_Bn=True, 
                 activation=nn.ReLU(inplace=True)):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.if_Bn = if_Bn
        if self.if_Bn:
            self.Bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    
    def forward(self, x):
        x = self.conv2d(x)
        if self.if_Bn:
            x = self.Bn(x)
        x = self.activation(x)
        return x

        
class W_Net(nn.Module):
    def __init__(self):
        super(W_Net, self).__init__()
        self.B1_C2 = nn.Sequential(*(list(vgg16bn_frop_fc.children())[0:6]))
        self.B2_C2 = nn.Sequential(*(list(vgg16bn_frop_fc.children())[6:13]))
        self.B3_C3 = nn.Sequential(*(list(vgg16bn_frop_fc.children())[13:23]))
        self.B4_C3 = nn.Sequential(*(list(vgg16bn_frop_fc.children())[23:33]))
        self.B5_C3 = nn.Sequential(*(list(vgg16bn_frop_fc.children())[33:43]))
        
        self.Connect_Module = ConnectLayer()
        
        self.Decoder_Block_1 = nn.Sequential(
            BasicConv2d(1024, 256, 1, 1, 0),
            BasicConv2d(256, 256, 3, 1, 1)
        )
        
        self.Decoder_Block_2 = nn.Sequential(
            BasicConv2d(512, 128, 1, 1, 0),
            BasicConv2d(128, 128, 3, 1, 1)
        )
        
        self.Decoder_Block_3 = nn.Sequential(
            BasicConv2d(256, 64, 1, 1, 0),
            BasicConv2d(64, 64, 3, 1, 1),
            BasicConv2d(64, 32, 3, 1, 1, if_Bn=False)
        )
        
        self.Reinforcement_Branch_Block_1 = nn.Sequential(
            BasicConv2d(1024, 256, 1, 1, 0),
            BasicConv2d(256, 256, 3, 1, 1)
        )
        
        self.Reinforcement_Branch_Block_2 = nn.Sequential(
            BasicConv2d(512, 128, 1, 1, 0),
            BasicConv2d(128, 128, 3, 1, 1)
        )
        
        self.Reinforcement_Branch_Block_3 = nn.Sequential(
            BasicConv2d(256, 64, 1, 1, 0),
            BasicConv2d(64, 64, 3, 1, 1),
            BasicConv2d(64, 32, 3, 1, 1, if_Bn=False),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        
        for block in [self.Decoder_Block_1, self.Decoder_Block_2, self.Decoder_Block_3, self.Reinforcement_Branch_Block_1, self.Reinforcement_Branch_Block_2, self.Reinforcement_Branch_Block_3, self.output_layer]:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        
    def forward(self, x):
        B1_C2_output = self.B1_C2(x)
        B2_C2_output = self.B2_C2(B1_C2_output)
        B3_C3_output = self.B3_C3(B2_C2_output)
        B4_C3_output = self.B4_C3(B3_C3_output)
        B5_C3_output = self.B5_C3(B4_C3_output)
        
        concated_encoded_map_1 = self.Connect_Module(
            decoded_input=B5_C3_output, 
            encoded_input=B4_C3_output
        )
        Decoder_Block_1_output = self.Decoder_Block_1(concated_encoded_map_1)
        Reinforcement_Branch_1_output = self.Reinforcement_Branch_Block_1(concated_encoded_map_1)
        concated_decoded_map_2 = self.Connect_Module(
            decoded_input=Decoder_Block_1_output,
            encoded_input=B3_C3_output
        )
        concated_reinforcement_map_2 = self.Connect_Module(
            decoded_input=Reinforcement_Branch_1_output,
            encoded_input=B3_C3_output
        )
        Decoder_Block_2_output = self.Decoder_Block_2(concated_decoded_map_2)
        Reinforcement_Branch_2_output=self.Reinforcement_Branch_Block_2(concated_reinforcement_map_2)
        concated_decoded_map_3 = self.Connect_Module(
            decoded_input=Decoder_Block_2_output,
            encoded_input=B2_C2_output
        )
        concated_reinforcement_map_3 = self.Connect_Module(
            decoded_input=Reinforcement_Branch_2_output,
            encoded_input=B2_C2_output
        )
        Decoder_Block_3_output = self.Decoder_Block_3(concated_decoded_map_3)
        Reinforcement_Branch_3_output=self.Reinforcement_Branch_Block_3(concated_reinforcement_map_3)
        element_wise_mul = torch.mul(Decoder_Block_3_output, Reinforcement_Branch_3_output)
        output = self.output_layer(element_wise_mul)
        return output
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tFunc

# OFDM Parameters
#S = 14  # Number of symbols
#Sp = 2  # Pilot symbol, 0 for none
#F = 72  # Number of subcarriers, including DC
class ResidualBlock(nn.Module):
    def __init__(self, S = 14, F = 72):
        super(ResidualBlock, self).__init__()

        #https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=(S,F-1))
        self.conv2d_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=(S,F-1))
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        z = self.layer_norm_1(inputs)
        z = tFunc.relu(z)
        z = self.conv2d_1(z)
        z = self.layer_norm_2(z)
        z = tFunc.relu(z)
        z = self.conv2d_2(z)
        
        # Skip connection
        z = z + inputs

        return z

class RXModel_2(nn.Module):

    def __init__(self, num_bits_per_symbol, S = 14, F = 72):
        super(RXModel_2, self).__init__()

        # Input convolution
        self.input_conv2d = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 3), padding='same')

        # Residual blocks
        self.res_block_1 = ResidualBlock(S = S, F = F)
        self.res_block_2 = ResidualBlock(S = S, F = F)
        self.res_block_3 = ResidualBlock(S = S, F = F)
        self.res_block_4 = ResidualBlock(S = S, F = F)
        self.res_block_5 = ResidualBlock(S = S, F = F)

        # Output conv
        self.output_conv2d = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding='same')

    def forward(self, inputs):
        y = inputs #[16, 14, 71]
   
        # Stack the tensors along a new dimension (axis 0)
        z = torch.stack([y.real, y.imag], dim=0) #[2, 16, 14, 71]
        z = z.permute(1, 0, 2, 3) #[16, 2, 14, 71]
        z = self.input_conv2d(z) #[16, 128, 14, 71]

        # Residual blocks
        z = self.res_block_1(z) #[16, 128, 14, 71]
        z = self.res_block_2(z) #[16, 128, 14, 71]
        z = self.res_block_3(z) #[16, 128, 14, 71]
        z = self.res_block_4(z)
        z = self.res_block_5(z) #[16, 128, 14, 71]
        z = self.output_conv2d(z) #[16, 6, 14, 71]

        # Reshape 
        z = z.permute(0,2, 3, 1) #[16, 14, 71, 6]
        z = nn.Sigmoid()(z) #[16, 14, 71, 6]

        return z

class ResModel_2D(nn.Module):

    def __init__(self, num_bits_per_symbol, num_ch=4, S = 14, F = 72):
        super(ResModel_2D, self).__init__()

        # Input convolution
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html (N, C_in, H, W) => (N, C_out, H, W)
        self.input_conv2d = nn.Conv2d(in_channels=num_ch, out_channels=128, kernel_size=(3, 3), padding='same')

        # Residual blocks
        self.res_block_1 = ResidualBlock(S = S, F = F)
        self.res_block_2 = ResidualBlock(S = S, F = F)
        self.res_block_3 = ResidualBlock(S = S, F = F)
        self.res_block_4 = ResidualBlock(S = S, F = F)
        self.res_block_5 = ResidualBlock(S = S, F = F)

        # Output conv
        self.output_conv2d = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding='same')

    def forward(self, inputs):
        y = inputs #[16, 4, 14, 71]
   
        # Stack the tensors along a new dimension (axis 0)
        #z = torch.stack([y.real, y.imag], dim=0) #[2, 16, 14, 71]
        #z = z.permute(1, 0, 2, 3) #[16, 2, 14, 71]
        z = self.input_conv2d(y) #[16, 128, 14, 71]

        # Residual blocks
        z = self.res_block_1(z) #[16, 128, 14, 71]
        z = self.res_block_2(z) #[16, 128, 14, 71]
        z = self.res_block_3(z) #[16, 128, 14, 71]
        z = self.res_block_4(z)
        z = self.res_block_5(z) #[16, 128, 14, 71]
        z = self.output_conv2d(z) #[16, 6, 14, 71]

        # Reshape 
        z = z.permute(0,2, 3, 1) #[16, 14, 71, 6]
        z = nn.Sigmoid()(z) #[16, 14, 71, 6]

        return z

class ResModel_simple1_2D(nn.Module):

    def __init__(self, num_bits_per_symbol, num_ch=4, S = 14, F = 72):
        super(ResModel_simple1_2D, self).__init__()

        # Input convolution
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html (N, C_in, H, W) => (N, C_out, H, W)
        self.layer_norm = nn.LayerNorm([128, S, F-1])
        self.input_conv2d = nn.Conv2d(in_channels=num_ch, out_channels=128, kernel_size=(1, 1), padding='same')
        # Output conv
        self.output_conv2d = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding='same')
        self.activation = nn.ReLU()
        self.linear=nn.Linear(in_features=6, out_features=6)

        # Residual blocks
        self.res_block_1 = ResidualBlock(S = S, F = F)
        self.res_block_2 = ResidualBlock(S = S, F = F)
        self.res_block_3 = ResidualBlock(S = S, F = F)
        self.res_block_4 = ResidualBlock(S = S, F = F)
        self.res_block_5 = ResidualBlock(S = S, F = F)

    def forward(self, inputs):
        y = inputs #[16, 4, 14, 71]
        z = self.input_conv2d(y) #[16, 128, 14, 71]
        z = self.layer_norm(z)
        z = self.activation(z)

        # Residual blocks
        z = self.res_block_1(z) #[16, 128, 14, 71]
        z = self.res_block_2(z) #[16, 128, 14, 71]
        z = self.res_block_3(z) #[16, 128, 14, 71]
        z = self.res_block_4(z)
        z = self.res_block_5(z) #[16, 128, 14, 71]
        z = self.output_conv2d(z) #[16, 6, 14, 71]
        
        # Reshape 
        z = z.permute(0,2, 3, 1) #[16, 14, 71, 6]
        z = self.linear(z) #[16, 14, 71, 6]
        z = nn.Sigmoid()(z) #[16, 14, 71, 6]
        return z

from transformers import Wav2Vec2Config
from wave2vec_lib import Wav2Vec2Encoder
class MyWave2vec(nn.Module):

    def __init__(self, num_bits_per_symbol, num_ch=4, S = 14, F = 72):
        super(MyWave2vec, self).__init__()
        configuration = Wav2Vec2Config()
        configuration.num_hidden_layers =3
        #print(configuration.hidden_size) #768
        configuration.hidden_size = 12*4 #head=12
        self.encoder = Wav2Vec2Encoder(configuration)
        self.linear=nn.Linear(in_features=num_ch, out_features=configuration.hidden_size)
        self.flatten = nn.Flatten(1,2)
        self.unflatten=nn.Unflatten(1, (S, F-1))
        self.outlinear=nn.Linear(in_features=configuration.hidden_size, out_features=num_bits_per_symbol)
        self.activation = nn.ReLU()
    
    def forward(self, inputs):
        feature_2d = inputs.permute(0,2,3, 1) #[1, 4, 14, 71]->[1, 14, 71, 4]
        feature_2d=self.linear(feature_2d) ##[1, 14, 71, 48]
        feature_2d=self.flatten(feature_2d) #[1, 14*71, 48] [1, 994, 48]
        encoder_outputs = self.encoder(
            feature_2d, #[1, 994, 48]
        ) #only one element: last_hidden_state
        hidden_states = encoder_outputs[0] #[1, 994, 48]
        output=self.unflatten(hidden_states) #[1, 14*71, 48] =>[1, 14, 71, 48]
        output=self.outlinear(output) #[1, 14, 71, 6]
        output = nn.Sigmoid()(output) #[1, 14, 71, 6]
        return output

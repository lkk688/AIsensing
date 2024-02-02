import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ofdmtrain_pytorch2 import OFDMDataset, MultiReceiver
from ofdmsim_pytorchlib import *
from wave2vec_lib import Wav2Vec2Model, Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection, Wav2Vec2Encoder
from transformers import Wav2Vec2Config

mycache_dir=os.path.join('E:',os.sep, 'Cache','huggingface')
os.environ['HF_HOME'] = mycache_dir
os.environ['HF_DATASETS_CACHE'] = mycache_dir

def testwave2vec():
    device, useamp=get_device(gpuid='0', useamp=False)

    # OFDM Parameters
    #Qm (int): Modulation order
    Qm = 6  # bits per symbol
    S = 14  # Number of symbols
    Sp = 2  # Pilot symbol, 0 for none
    F = 72  # Number of subcarriers, including DC
    #5-50 0331exp
    #-10 20 exp0201
    #-10 40 exp0201b
    train_data = OFDMDataset(Qm=Qm, S=S, Sp=Sp, F=F, ch_SINR_min=-10, ch_SINR_max=40, training=True)
    onebatch = train_data[0]
    val_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    onebatch = next(iter(val_loader))
    rx_samples = onebatch['samples']
    feature_2d = onebatch['feature_2d']
    data_labels = onebatch['labels']
    print(f"Sample batch shape: {rx_samples.size()}") #[1, 2078]
    print(f"Feature batch shape: {feature_2d.size()}") #[1, 4, 14, 71] #BCHW, 4 is C
    print(f"Labels batch shape: {data_labels.size()}") #[1, 14, 71, 6]

    # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
    configuration = Wav2Vec2Config()
    # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    model = Wav2Vec2Model(configuration)

    feature_extractor = Wav2Vec2FeatureEncoder(configuration)
    feature_projection = Wav2Vec2FeatureProjection(configuration)

    input_values = rx_samples.real #[1, 2078]
    extract_features = feature_extractor(input_values) #[1, 86699] -> [1, 512, 270]
    extract_features = extract_features.transpose(1, 2) #[1, 270, 512]

    hidden_states, extract_features = feature_projection(extract_features) #[1, 270, 768] 

    print(configuration.num_hidden_layers) #12
    configuration.num_hidden_layers =1
    print(configuration.hidden_size) #768
    configuration.hidden_size = 12*4 #head=12
    encoder = Wav2Vec2Encoder(configuration)

    #feature_2d [1, 4, 14, 71]
    feature_2d = feature_2d.permute(0,2,3, 1) #[1, 4, 14, 71]->[1, 14, 71, 4]
    linear=nn.Linear(in_features=4, out_features=configuration.hidden_size)
    feature_2d=linear(feature_2d) ##[1, 14, 71, 48]
    flatten = nn.Flatten(1,2)
    feature_2d=flatten(feature_2d) #[1, 14*71, 48] [1, 994, 48]
    encoder_outputs = encoder(
            feature_2d, #[1, 994, 48]
        ) #only one element: last_hidden_state
    hidden_states = encoder_outputs[0] #[1, 994, 48]
    unflatten=nn.Unflatten(1, (14, 71))
    output=unflatten(hidden_states) #[1, 14*71, 48] =>[1, 14, 71, 48]
    outlinear=nn.Linear(in_features=configuration.hidden_size, out_features=6)
    output=outlinear(output) #[1, 14, 71, 6]
    print(output.shape) #[1, 14, 71, 6]
    
if __name__ == '__main__':
    testwave2vec()
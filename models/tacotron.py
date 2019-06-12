# coding: utf-8
import torch
from torch import nn
from math import sqrt
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG

import time

class Tacotron(nn.Module):
    def __init__(self,
                 num_chars=61,
                 embedding_dim=256,
                 linear_dim=1025,
                 mel_dim=80,
                 r=5,
                 padding_idx=None, 
                 memory_size=5,
                 attn_windowing=False):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(
            num_chars, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(256, mel_dim, r, memory_size, attn_windowing)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim),
            nn.Sigmoid())

    def forward(self, characters, mel_specs=None, mask=None):
        
        batch_size = characters.size(0)
        
        # Get the embeddings of our input 
        embedded_inputs = self.embedding(characters)

        # Encode the inputs
        # Shape : batch_size x number_of_phonemes x (embedding_dimension / 2)
        encoder_outputs = self.encoder(embedded_inputs)

        # Decode the encoded data
        # Shape : batch_size x number_of_phonemes x embedding_dimension
        mel_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, mask)

        # Reshape the data so each representation has desired mel spectogram frame dimension
        mel_outputs = mel_outputs.view(batch_size, -1, self.mel_dim)

        # Pass data to the postnet CBHG
        # Shape : batch_size x dim x mel spectogram frame dimension
        postnet_output = self.postnet(mel_outputs)

        # Pass data through the last linear layers
        # Shape : batch_size x dim x embedding_dimension
        linear_outputs = self.last_linear(postnet_output)   

        # Linear output shape : batch_size x dim x linear dimension
        return mel_outputs, linear_outputs, alignments, stop_tokens    

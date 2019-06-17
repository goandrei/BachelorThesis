import io
import os
import sys
import time

#append the parent directory as a search path
file_directory = os.path.dirname(__file__)
sys.path.append(file_directory + '/..')

import numpy as np
import torch

#in-house imports
from models.tacotron import Tacotron
from utils.audio import AudioProcessor
from utils.generic_utils import load_config
from utils.text import phoneme_to_sequence, phonemes, symbols, text_to_sequence

"""
Class used by the Flask server in order to test the inference
Contains 3 functions:
-load_model : loads the model and keeps a copy of it as a parameter
-save_wav   : saves a given waveform
-tts        : gets a text and returns the model's prediction
"""
class Synthesizer(object):

    """
    Summary:
        Config is loaded and the model from the given path is loaded and prepared for inference.

    Parameters:
        @model_path = model's file directory path
        @model_name = model's file name
        @model_config = config's file name
        @use_cuda = GPU flag
    """
    def load_model(self, model_path, model_name, model_config, use_cuda):

        #build the config's path
        model_config = os.path.join(model_path, model_config)

        #build the model's path
        model_file = os.path.join(model_path, model_name)
        print(" > Loading model ...")
        print(" | > Model config path: ", model_config)
        print(" | > Model file path: ", model_file)

        config = load_config(model_config)
        self.use_cuda = use_cuda
        self.use_phonemes = config.use_phonemes
        self.ap = AudioProcessor(**config.audio)

        if self.use_phonemes:
            self.input_size = len(phonemes)
            self.input_adapter = lambda sen: phoneme_to_sequence(sen, [config.text_cleaner], config.phoneme_language)
        else:
            self.input_size = len(symbols)
            self.input_adapter = lambda sen: text_to_sequence(sen, [config.text_cleaner])

        self.model = Tacotron(num_chars=config['num_chars'], embedding_dim=config['embedding_size'], 
                              linear_dim=self.ap.num_freq, mel_dim=self.ap.num_mels, r=config['r'])
        
        #load model state
        if use_cuda:
            cp = torch.load(model_file)
        else:
            cp = torch.load(model_file, map_location=lambda storage, loc: storage)
        
        #load the model
        self.model.load_state_dict(cp['model'])

        #if cuda is enabled & available move tensors to GPU
        if use_cuda:
            self.model.cuda()

        #disables normalization techniques present in code
        self.model.eval()

    """
    Summary:
        Saves the wav at the given path

    Parameters:
        @wav = wav array
        @path = destination path
    """
    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    """
    Summary:
        Gets an input, prepares it for the model and returns the predicted output.

    Parameters:
        @text = input sentence 
    """
    def tts(self, text, gl_mode=None):                

        wavs = []
        
        #split the input in sentences
        for sen in text.split('.'):

            if len(sen) < 3:
                continue

            sen = sen.strip()
            sen += '.'
            #print('Input : {}'.format(sen)) 

            #character => phonem => index
            seq = np.array(self.input_adapter(sen))
            
            #numpy to pytorch array
            chars_var = torch.from_numpy(seq).unsqueeze(0).long()

            if self.use_cuda: 
                chars_var = chars_var.cuda()

            #begin the inference
            mel_out, linear_out, alignments, stop_tokens = self.model.forward(chars_var)

            #move output tensor to cpu
            linear_out = linear_out[0].data.cpu().numpy() 
            t = time.time()      
            wav = self.ap.inv_spectrogram(linear_out.T, gl_mode)   
            t = time.time() - t
            wavs += list(wav)
            wavs += [0] * 10000

        out = io.BytesIO() 
        self.save_wav(wavs, out)
        self.save_wav(wavs, 'gla.wav')
        return out   

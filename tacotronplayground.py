# %load_ext autoreload
# %autoreload 2
import os
import sys
import io
import torch 
import time
import numpy as np
from collections import OrderedDict
from matplotlib import pylab as plt

# %pylab inline
#rcParams["figure.figsize"] = (16,5)
sys.path.append('/home/andrei/Desktop/TTS-db7f3d36e7768f9179d42a8f19b88c2c736d87eb/')
sys.modules.keys()

import librosa
import librosa.display

from models.tacotron import Tacotron 
from layers import *
from utils.data import *
from utils.audio import AudioProcessor
from utils.generic_utils import load_config
from utils.text import text_to_sequence

import IPython
from IPython.display import Audio
from utils import *
from synthesis import create_speech

def tts(model, text, CONFIG, use_cuda, ap, figures=True):
    t_1 = time.time()
    waveform, alignment, spectrogram, stop_tokens = create_speech(model, text, CONFIG, use_cuda, ap) 
    print(" >  Run-time: {}".format(time.time() - t_1))
    if figures:                                                                                                         
        visualize(alignment, spectrogram, stop_tokens, CONFIG)                                                                       
    IPython.display.display(Audio(waveform, rate=CONFIG.sample_rate))  
    return alignment, spectrogram, stop_tokens

# Set constants
ROOT_PATH = ''
MODEL_PATH = ROOT_PATH + 'best_model.pth.tar'
CONFIG_PATH = ROOT_PATH + 'config.json'
OUT_FOLDER = ROOT_PATH + ''
CONFIG = load_config(CONFIG_PATH)
use_cuda = False

# load the model
model = Tacotron(61, CONFIG.embedding_size, CONFIG.audio['num_freq'], CONFIG.audio['num_mels'], CONFIG.r)

# load the audio processor

ap = AudioProcessor(CONFIG.audio['sample_rate'], CONFIG.audio['num_mels'], CONFIG.audio['min_level_db'],
                    CONFIG.audio['frame_shift_ms'], CONFIG.audio['frame_length_ms'], CONFIG.audio['preemphasis'],
                    CONFIG.audio['ref_level_db'], CONFIG.audio['num_freq'], CONFIG.audio['power'], griffin_lim_iters=30)         


# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

new=list(cp['model'].items())

my_model_kvpair=model.state_dict()
count=0
for key,value in my_model_kvpair.items():
    layer_name,weights=new[count]      
    if weights.shape == my_model_kvpair[key].shape:
        continue
    #print(layer_name)
    #print(weights.shape)
    #print(key)
    #print(my_model_kvpair[key].shape)
    #my_model_kvpair[key].view(weights.shape)
    #print('')
    count+=1

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

"""### EXAMPLES FROM TRAINING SET"""

"""
import pandas as pd
df = pd.read_csv('/data/shared/KeithIto/LJSpeech-1.0/metadata_val.csv', delimiter='|')

sentence = df.iloc[175, 1]
print(sentence)
model.decoder.max_decoder_steps = 250
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)
"""
"""### Comparision with https://mycroft.ai/blog/available-voices/"""

sentence =  "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
model.decoder.max_decoder_steps = 250
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap, figures=True)

sentence = "Be a voice,not an echo."  # 'echo' is not in training set. 
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "The human voice is the most perfect instrument of all."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "I'm sorry Dave. I'm afraid I can't do that."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "This cake is great. It's so delicious and moist."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

"""### Comparison with https://keithito.github.io/audio-samples/"""

sentence = "Generative adversarial network or variational auto-encoder."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "Scientists at the CERN laboratory say they have discovered a new particle."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "here’s a way to measure the acute emotional intelligence that has never gone out of style."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "President Trump met with other leaders at the Group of 20 conference."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)

sentence = "The buses aren't the problem, they actually provide a solution."
align, spec, stop_tokens = tts(model, sentence, CONFIG, use_cuda, ap)
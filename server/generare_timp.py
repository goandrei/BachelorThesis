from synthesizer import Synthesizer
from utils.generic_utils import load_config
import time
import matplotlib.pyplot as plt

import os
os.chdir('/home/andrei/BachelorThesis/server')

config = load_config('config.json')
synthesizer = Synthesizer()
synthesizer.load_model(model_path=config['model_path'], model_name=config['model_name'],
                       model_config=config['model_config'], use_cuda=config['use_cuda'])

short_inputs = ['I am a test.',
          'This is a short input.',
          'I must be faster when speaking.',
          'I love to speak to you.',
          'Machines have become increasingly capable.']

#source: https://en.wikipedia.org/wiki/Artificial_intelligence
medium_inputs = ['Artificial intelligence can be classified into three different types of systems.',
                 'I am a medium input and I wouldnt be so long and annoying if I would want to.',
                 'Its nice to be important but is more important to be nice thats what a wise man said.',
                 'I searched so much for medium sentences so I ran out of ideas so now I am writing random stuff.',
                 'This enables even young children to easily make inferences like If I roll this pen off a table, it will fall on the floor.']

#source: https://en.wikipedia.org/wiki/Artificial_intelligence
#        https://en.wikipedia.org/wiki/Machine_learning
long_inputs = ['Many researchers predict that such "narrow AI" work in different individual domains will eventually be incorporated into a machine with artificial general intelligence.',
               'When access to digital computers became possible in the middle 1950s, AI research began to explore the possibility that human intelligence could be reduced to symbol manipulation.',
               'Unlike Simon and Newell, John McCarthy felt that machines did not need to simulate human thought, but should instead try to find the essence of abstract reasoning and problem-solving.',
               'In supervised learning, the algorithm builds a mathematical model from a set of data that contains both the inputs and the desired outputs.',
               'Machine learning is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions.']

def global_timer():
    total_time = 0
    for idx, short_input in enumerate(short_inputs):
        t1 = time.time()
        data = synthesizer.tts(short_input)
        t2 = time.time()
        diff = t2 - t1
        print('Input ', idx, ' : ', diff)
        total_time += diff

    short_mean_time = total_time / 5
    print('Short input mean time : ', short_mean_time)
    print()

    total_time = 0
    for idx, medium_input in enumerate(medium_inputs):
        t1 = time.time()
        data = synthesizer.tts(medium_input)
        t2 = time.time()
        diff = t2 - t1
        print('Input ', idx, ' : ', diff)
        total_time += diff

    medium_mean_time = total_time / 5
    print('Medium input mean time : ', medium_mean_time)
    print()

    total_time = 0
    for idx, long_input in enumerate(long_inputs):
        t1 = time.time()
        data = synthesizer.tts(long_input)
        t2 = time.time()
        diff = t2 - t1
        print('Input ', idx, ' : ', diff)
        total_time += diff

    long_mean_time = total_time / 5
    print('Long input mean time : ', long_mean_time)

    times = [short_mean_time, medium_mean_time, long_mean_time]

    plt.bar([1,2,3], height=times)
    plt.xticks([1, 2, 3], ['Intrare scurta(5)', 'Intrare medie(15)', 'Intare lunga(30)'])
    plt.show()

def module_timer():

    times = [0,0,0]
    for idx, short_input in enumerate(short_inputs):
        data, t = synthesizer.tts(short_input)
        print(idx, ': ', t)
        times[0] += t

    for idx, medium_input in enumerate(medium_inputs):
        data, t = synthesizer.tts(medium_input)
        print(idx, ': ', t)
        times[1] += t

    for idx, long_input in enumerate(long_inputs):
        data, t = synthesizer.tts(long_input)
        print(idx, ': ', t)
        times[2] += t

    plt.bar([0.75,1.75,2.75], height=times)
    plt.xticks([0.75, 1.75, 2.75], ['Intrare scurta(5)', 'Intrare medie(15)', 'Intare lunga(30)'])
    plt.show()

if __name__ == '__main__':

    #module_timer()
    data = synthesizer.tts('hi there how are you')

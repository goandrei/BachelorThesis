#!flask/bin/python
import argparse
import torch
from synthesizer import Synthesizer
from utils.generic_utils import load_config
from flask import Flask, Response, request, render_template, send_file

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_path', type=str, help='path to config file for training')
args = parser.parse_args()

config = load_config('server/config.json')
app = Flask(__name__)
synthesizer = Synthesizer()
synthesizer.load_model(config.model_path, config.model_name,
                       config.model_config, config.use_cuda)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')  
    synthesizer.ap.frame_shift_ms        = int(request.args.get('shift'))
    synthesizer.ap.griffin_lim_iters     = int(request.args.get('iter'))
    synthesizer.ap.griffin_lim_algorithm = request.args.get('gl')
    synthesizer.ap._stft_parameters()   
    synthesizer.ap.set_gl_algorithm()

    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')                    
 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=config.port)

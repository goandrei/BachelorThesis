import os
import librosa
import pickle
import copy
import numpy as np
from pprint import pprint
from scipy import signal, io
from utils.visual import plot_spectrogram

class AudioProcessor(object):
    def __init__(self,
                 bits=None,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=None,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 **kwargs):

        print(" > Setting up Audio Processor...")

        self.bits = bits
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = 0 if mel_fmin is None else mel_fmin
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        members = vars(self)
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _build_mel_basis(self, ):
        n_fft = (self.num_freq - 1) * 2
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        if self.signal_norm:
            S_norm = ((S - self.min_level_db) / - self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm :
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm) 
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                    self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x): 
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram, gl_mode=None):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            if gl_mode == 'admm':
                return self.apply_inv_preemphasis(self._admm_griffin_lim(S**self.power))
            if gl_mode == 'gla':
                return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
            if gl_mode == 'fgla':
                return self.apply_inv_preemphasis(self._fast_griffin_lim(S**self.power))
            if gl_mode == 'fgla2':
                return self.apply_inv_preemphasis(self._fast_griffin_lim2(S**self.power))
            if gl_mode == 'mfgla':
                return self.apply_inv_preemphasis(self._mod_fast_griffin_lim(S**self.power))

            return self.apply_inv_preemphasis(self._admm_griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def _griffin_lim(self, S):
        print('gla')
        # build the initial phase array
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))

        # build some complex numbers using spectogram information as real part(magnitude) 
        # we will firstly use a random imaginary part as phase, then update it each iteration
        S_complex = np.abs(S).astype(np.complex)
        
        # compute the first time-series
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            # apply STFT to get back the spectogram, then compute the angle(imaginary part)
            # and compute the new phase
            angles = np.exp(1j * np.angle(self._stft(y)))
            
            # apply inverse STFT to get the time-series
            y = self._istft(S_complex * angles)
        return y

    def _fast_griffin_lim(self, S):
        print('fast')
        # build the initial phase array
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))

        # build some complex numbers using spectogram information as real part(magnitude) 
        # we will firstly use a random imaginary part as phase, then update it each iteration
        S_complex = np.abs(S).astype(np.complex)
        alfa = 0.98
        # compute the first time-series
        t0 = self._stft(self._istft(S_complex * angles))
        for i in range(self.griffin_lim_iters):
            t1 = self._stft(self._istft(S_complex * angles))
            angles = np.exp(1j * np.angle(t1 + alfa*(t1 - t0)))
            t1 = t0

        return self._istft(S_complex * angles)

    # source: https://github.com/rbarghou/pygriffinlim
    def _fast_griffin_lim2(self, S):
        print('fast2')
        _M = S
        alpha = 0.1
        aprox_signal = None
        for k in range(self.griffin_lim_iters):
            if aprox_signal is None:
                _P = np.random.randn(*_M.shape)
            else:
                _D = self._stft(aprox_signal)
                _P = np.angle(_D)

            _D = _M * np.exp(1j * _P)
            _M = S + (alpha * np.abs(_D))
            aprox_signal = self._istft(_D)

        return aprox_signal

    def _mod_fast_griffin_lim(self, S):
        print('mod_fast')

        _M = S
        aprox_signal = None
        for k in range(self.griffin_lim_iters):
            if aprox_signal is None:
                _P = np.random.randn(*_M.shape)
            else:
                _D = self._stft(aprox_signal)
                _P = np.angle(_D)

            _D = _M * np.exp(1j * _P)
            alpha = np.random.normal(0.1, 0.4)
            _M = S + (alpha * np.abs(_D))
            aprox_signal = self._istft(_D)

        return aprox_signal

    def mysign(self,x):
        return np.exp(1j * np.angle(x))

    def _admm_griffin_lim(self, S, rho = 0.1):
        print('admm')
        S = np.abs(S)
        spec_shape = S.shape
        z = S * np.exp(2 * np.pi * 1j * np.random.rand(spec_shape[0],spec_shape[1]))
        u = np.zeros(spec_shape)

        for i in range(self.griffin_lim_iters):
            x = S * self.mysign(z-u)
            v = x + u
            z = (rho * v + self._stft(self._istft(v))) / (1 + rho)
            u = u + x - z
        return self._istft(x)

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.1 sec margin """
        margin = int(self.sample_rate * 0.1)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=40, frame_length=1024, hop_length=256)[0]

    # WaveRNN repo specific functions
    # def mulaw_encode(self, wav, qc):
    #     mu = qc - 1
    #     wav_abs = np.minimum(np.abs(wav), 1.0)
    #     magnitude = np.log(1 + mu * wav_abs) / np.log(1. + mu)
    #     signal = np.sign(wav) * magnitude
    #     # Quantize signal to the specified number of levels.
    #     signal = (signal + 1) / 2 * mu + 0.5
    #     return signal.astype(np.int32)

    # def mulaw_decode(self, wav, qc):
    #     """Recovers waveform from quantized values."""
    #     mu = qc - 1
    #     # Map values back to [-1, 1].
    #     casted = wav.astype(np.float32)
    #     signal = 2 * (casted / mu) - 1
    #     # Perform inverse of mu-law transformation.
    #     magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
    #     return np.sign(signal) * magnitude

    def load_wav(self, filename, encode=False):
        x, sr = librosa.load(filename, sr=self.sample_rate)
        if self.do_trim_silence:
            x = self.trim_silence(x)
        # sr, x = io.wavfile.read(filename)
        assert self.sample_rate == sr
        return x

    def encode_16bits(self, x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def quantize(self, x):
        return (x + 1.) * (2**self.bits - 1) / 2

    def dequantize(self, x):
        return 2 * x / (2**self.bits - 1) - 1

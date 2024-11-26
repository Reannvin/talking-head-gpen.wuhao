import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from hparams import hparams as hp
import torchaudio
import torch
from torchvision import transforms

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        # k = 0.97
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram2(wav):
    # import ipdb;ipdb.set_trace() 
    y = librosa.effects.preemphasis(wav, coef=hp.preemphasis)
    print(get_hop_size(), hp.win_size)
    # return y
    D = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)
    # D = _stft(preempha)
    # S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    # return _normalize(S)
    spectogram = np.abs(D)
    # return spectogram 
    # _linear_to_mel
    assert hp.fmax <= hp.sample_rate // 2
    _mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)
    S = np.dot(_mel_basis, spectogram)
    # return S
    # _amp_to_db
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    S = 20 * np.log10(np.maximum(min_level, S))
    S = S - hp.ref_level_db

    # return S
    return _normalize(S)



class iAmplitudeToDB(torch.nn.Module):
    def __init__(self, min_level_db, ref_level_db, max_abs_value):
        super(iAmplitudeToDB, self).__init__()
        
        self.min_level = torch.exp(min_level_db / 20 * torch.log(torch.tensor([10])))
        self.ref_level_db = ref_level_db
        self.max_abs_value = max_abs_value
        self.min_level_db = min_level_db

        
    def _normalize(self, S):
        return torch.clip((2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                            -self.max_abs_value, self.max_abs_value)
        

    def forward(self, x):
        
        S = 20 * torch.log10(torch.clamp(x, min=self.min_level.to(x)))
        S = S - self.ref_level_db
        
        return self._normalize(S)

def get_audio_transforms():
    transform1 = torchaudio.transforms.Preemphasis()
    n_fft = 800
    hop_length =  200 
    win_length = 800

    transform2 = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, 
                                             pad_mode='constant', power=1)
    sr = 16000
    n_mels = 80
    f_min = 55
    f_max = 7600
    transform3 = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sr, f_min=f_min, f_max=f_max, n_stft=401, 
                                                norm='slaney', mel_scale='slaney')


    transform4 = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length= win_length, hop_length=hop_length, 
                                                      f_min=f_min, f_max=f_max, pad=0, n_mels=n_mels,  power= 1.0, normalized= False, 
                                                      center=True, pad_mode='constant', norm='slaney', mel_scale='slaney')#.to('cuda')

    transform5 =  iAmplitudeToDB(min_level_db=-100, ref_level_db=20, max_abs_value=4)

    transform = transforms.Compose([transform1, transform4, transform5])

    return transform


def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

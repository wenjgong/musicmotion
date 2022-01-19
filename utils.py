import sys
import librosa
import numpy as np
import soundfile as sf
import functools
import torch
import glob
from torch.nn.functional import cosine_similarity
fps = 25

def logme(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        print('\n-----------------\n')
        print('   MODEL: {}'.format(f.__name__.upper()))
        print('\n-----------------\n')
        return f(*args, **kwargs)
    return wrapped


class ProgressBar:
    """Progress bar
    
    """
    def __init__ (self, valmax, maxbar, title):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title
        print ('')

    def update(self, val, avg_loss=0):
        # format
        if val > self.valmax: val = self.valmax

        # process
        perc  = round((float(val) / float(self.valmax)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)

        # render
        if avg_loss:
            # out = '\r %20s [%s%s] %3d / %3d  cost: %.2f  r_loss: %.0f  l_loss: %.4f  clf_loss: %.4f' % (
            out = '\r %20s [%s%s] %3d / %3d  loss: %.5f' % (
                self.title, 
                '=' * bar, ' ' * (self.maxbar - bar), 
                val, 
                self.valmax, 
                avg_loss, 
                )
        else:
            out = '\r %20s [%s%s] %3d / %3d ' % (self.title, '=' * bar, ' ' * (self.maxbar - bar), val, self.valmax)

        sys.stdout.write(out)
        sys.stdout.flush()


def pad(l, sr):
    # 0-Pad 10 sec at fs hz and add little noise
    z = np.zeros(10*sr, dtype='float32')
    z[:l.size] = l
    z = z + 5*1e-4*np.random.rand(z.size).astype('float32')
    return z
def find_files(root_dir, query="DANCE**.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for filename in glob.iglob(f'{root_dir}/**/{query}', recursive=True):
       # if filename.split('\\')[1].split('_')[1] == 'W':
       #     continue
       files.append(filename)

    # for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
    #     for filename in fnmatch.filter(filenames, query):
    #         files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def compute_spectrogram(filename, start, duration, sr=12800, n_mels=96):
    # zero pad and compute log mel spec
    try:
        audio, sr = librosa.load(filename, sr=sr, offset=start/fps,duration=(duration-1)/fps,res_type='kaiser_fast')
    except:
        audio, o_sr = sf.read(filename)
        audio = librosa.core.resample(audio, o_sr, sr)
    try:
        x = pad(audio, sr)
    except ValueError:
        x = audio

    x_stft = librosa.stft(x, n_fft=1024, hop_length=512,
                          win_length=1024, window='hann', pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 80
    fmax = 7600
    mel_basis = librosa.filters.mel(sr, 1024, 96, fmin, fmax)

    return np.log10(np.maximum(1e-10, np.dot(spc, mel_basis.T)))

    # audio_rep = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=512, n_fft=1024, n_mels=n_mels, power=1.)
    # audio_rep = np.log(audio_rep + np.finfo(np.float32).eps)
    #
    # audio_mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=96,hop_length=512,n_fft=1024,power=1.)
    # #不能log，因为MFCC中有负数
    # #audio_mfcc = np.log(audio_mfcc + np.finfo(np.float32).eps)
    #
    # return audio_mfcc


def return_spectrogram_max_nrg_frame(spectrogram):
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idx_max_nrg = np.argmax(np.sum(np.sum(frames, axis=0), axis=0))
    return frames[:,:,idx_max_nrg]


def return_spectrogram_3_max_nrg_frames(spectrogram):
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idxes_max_nrg = (-np.sum(np.sum(frames, axis=0), axis=0)).argsort()[:3]
    return frames[:,:,idxes_max_nrg]


def spectrogram_to_audio(filename, y, sr=22000):
    y = np.exp(y)
    x = librosa.feature.inverse.mel_to_audio(y, sr=sr, n_fft=1024, hop_length=512, power=1.)
    librosa.output.write_wav(filename, x, sr)


def kullback_leibler(y_hat, y):
    """Generalized Kullback Leibler divergence.
    :param y_hat: The predicted distribution.
    :type y_hat: torch.Tensor
    :param y: The true distribution.
    :type y: torch.Tensor
    :return: The generalized Kullback Leibler divergence\
             between predicted and true distributions.
    :rtype: torch.Tensor
    """
    return (y * (y.add(1e-5).log() - y_hat.add(1e-5).log()) + (y_hat - y)).sum(dim=-1).mean()
def LogCoshLoss(y, y_):
    ey_t = y - y_
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def XTanhLoss(y, y_):
    ey_t = y - y_
    return torch.mean(ey_t * torch.tanh(ey_t))

def XSigmoidLoss(y, y_):
    ey_t = y - y_
    return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


def embeddings_to_cosine_similarity_matrix(z):
    """Converts a a tensor of n embeddings to an (n, n) tensor of similarities.
    """
    cosine_similarity = torch.matmul(z, z.t())
    embedding_norms = torch.norm(z, p=2, dim=1)
    embedding_norms_mat = embedding_norms.unsqueeze(0)*embedding_norms.unsqueeze(1)
    cosine_similarity = cosine_similarity / (embedding_norms_mat)
    return cosine_similarity


def contrastive_loss(z_audio, z_tag, t=1):
    """Computes contrastive loss following the paper:
        A Simple Framework for Contrastive Learning of Visual Representations
        https://arxiv.org/pdf/2002.05709v1.pdf
        TODO: make it robust to NaN (with low values of t it happens). 
        e.g Cast to double float for exp calculation.
    """
    z = torch.cat((z_audio, z_tag), dim=0)
    s = embeddings_to_cosine_similarity_matrix(z)
    N = int(s.shape[0]/2)
    s = torch.exp(s/t)
    try:
        s = s * (1 - torch.eye(len(s), len(s)).cuda())
        # s[range(len(s)), range(len(s))] = torch.zeros((len(s),)).cuda()
    except AssertionError:
        s = s * (1 - torch.eye(len(s), len(s)))
    denom = s.sum(dim=-1)
    num = torch.cat((s[:N,N:].diag(), s[N:,:N].diag()), dim=0)
    return torch.log((num / denom) + 1e-5).neg().mean()

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

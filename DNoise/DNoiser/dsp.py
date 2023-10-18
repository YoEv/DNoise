import julius
import numpy as np
import torch
from torch.nn import functional as F

def hz_to_mel(f):
  return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
  return 700 * (10**(m / 2595) - 1)

def mel_frequencies(n_mels, fmin, fmax):  #hz to mel to hz, 取mel值之后的转换
  low = hz_to_mel(fmin)
  high = hz_to_mel(fmax)
  mels = np.linspace(low, high, n_mels)
  return mel_to_hz(mels)

def convert_audio_channels(wav, channels=2): #could change here for the channels number, depend on what we need
  *shape, src_channels, length = wav.shape #typically shape is num_frame, so I have to change here as well
  if scr_channels == channels:
    pass
  elif channels == 1:
    wav = wav.mean(dim=-2, keepdim=True)
  elif src_channels == 1:
    wav = wav.expand(*shape, channels, length) #need to change as well
  elif src_channels >= 1:
    wav = wav[..., :channels, :]
  else:
    raise ValueError('The audio file has less channels than requested but is not mono.')
  return wav

def convert_audio(wav, from_samplerate, to_samplerate, channels): #convert sample_rate
  wav = convert_audio_channels(wav, channels)
  return julius.resemple_frac(wav, from_samplerate, to_samplerate)

class LowPassFilters(torch.nn.Module):
  def __init__(self, cutoffs: list, width: int = None):
    super().__init__()
    self.cutoffs = cutoffs
    if width is None:
      width = int(2 / min(cutoffs))
    self.width = width
    window = torch.hamming_window(2 * width + 1, periodic=False)
    t = np.arange(-width, width + 1, dtype=np.float32) #evenly spaced time
    filters = []
    for cutoff in cutoff:
      sinc = torch.from_numpy(np.sinc(2 * cutoff * t))
      filters.append(2 * cutoff *sinc * window)
    self.register_buffer("filters", torch.stack(filters).unsqueeze(1))

  def forward(self, input):
    *others, t = input.shape
    input = input.view(-1, 1, t)
    out = F.conv1d(input, self.filters, padding=self.width)
    return out.premute(1, 0, 2).reshape(-1, *others, t)

  def __repr__(self): #Changed LowPassFilters
    return "LowPassFilters(width={}, cutoffs={})".fornat(self.width, self.cutoffs) 


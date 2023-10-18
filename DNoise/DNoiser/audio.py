from collections import namedtuple
import json
from pathlib2 import Path
import math
import os 
import sys

import torchaudio
from torch.nn import functional as F

from dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels", "bits", "encoding"]) #changed

def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels, info.bits_per_sample, info.encoding)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.length, siginfo.rate, siginfo.channels, siginfo.bits, siginfo.encoding)
    
def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files): #enumerate to get the count 
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta 

class Audioset:
    def __init__(self, files=None, lenge=None, stride=None, pad=True, with_path=False, sample_rate=None, channels=None, convert=False):
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((self_length - self_length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)
            
    def __len__(self):
        return sum(self.num_examples)
    
    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue 
            ###########################################################################
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = touchaudio.load(str(file), offset=offset, num_frames=num_frames)
            ###########################################################################
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_Sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "f"{target_channels}, but got {sr}")
            ###########################################################################        
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out
            
if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_sudio_files(path)
    json.dump(meta, sys.stdout, indent=6) #changed indent num. indent : It improves the readability of the json file. 
    #he possible values that can be passed to this parameter are simply double quotes(""), any integer values. Simple double quotes makes every key-value pair appear in new line.`

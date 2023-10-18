import json
import logging
import os
import re
from audio import Audioset

logger = logging.getLogger(__name__)

def match_dns(noisy, clean):
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'field_(\d+).wav$', path)
        if match is None:
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    exrta_clean = []
    copied = list(clean)
    clean[:] = []
    
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy
    
def match_files(noisy, clean, matching="sort"):
    if matching == "dns":
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")
        
class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None, pad=True, sample_rate=None):
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)
            
        match_files(noise, clean, matching) 
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw) #DO I HAVE to add all features here????
        self.noisy_set = Audioset(noisy, **kw)
        
    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]
    
    def __len__(self):
        return len(self.noisy_set)



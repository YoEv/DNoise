# DNoise


## Notebook of DNoise Processor 
### Downloaded Packages 
- Json `conda install -c jmcmurray json`
- Pathlib2 `conda install -c anaconda pathlib2`
- os `conda install -c jmcmurray os`
- collections-extensions `pip install collections-extended`
- torchaudio `conda install -c pytorch torchaudio`
- PyTorch `conda install -c pytorch pytorch`
- julius `pip install julius`


### Learnt Python Files
#### Datasetup
- `audio.py`
- `data.py`
- `dsp.py` - `hz_to_mel`, `mel_to_hz`, `mel_frequencies(n_mels, fmin, fmax)`, NEED TO CHANGE `convert_audio_channels`,   `def __repr__(self):` #Changed LowPassFilters
- `utils.py`
#### Model Built
- `resample.py`
  - Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984. 

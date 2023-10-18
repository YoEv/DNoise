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
- `audio.py` - for processing audio data sets, as well as some helper functions and named tuples for processing audio file information。
- `data.py` - Process noisy and clean audio data sets, match noise and clean audio data, sort, read data stored in JSON files, match noise and clean audio into pairs, obtain length information of the data set.
- `dsp.py` - `hz_to_mel`, `mel_to_hz`, `mel_frequencies(n_mels, fmin, fmax)`, NEED TO CHANGE `convert_audio_channels`,   `def __repr__(self):` #Changed LowPassFilters
- `utils.py`
- **Collected Data**
- <img width="633" alt="LogicDNoise" src="https://github.com/YoEv/DNoise/assets/98185075/4a60bb82-c259-4563-bf14-fd0d0aeaae64">
- <img width="672" alt="AudioFile" src="https://github.com/YoEv/DNoise/assets/98185075/ca7462db-876b-4085-a09c-8ee3f5173a16">



#### Setup Distributed Training Env
- `distrib.py`
  - This is a little bit wierd, it only check whether the `world_size == 1`, and it always returns.
  - Also, I changed in the Colab file of this to AMD with ROMs, and kept the CUDA one there. 
- `executor.py`
  - Use hydra to control the training configurate to better locate and manage different modules while training.
  - Run on CUDA GPU - Consider to change or not neccessary. - It takes more work to do so. 

#### Model Built
- `resample.py`
  - Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
  - 1-dimensional convolution operation. `out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)`
- `demucs.py`
  - .
  - `def valid_length` to make sure that under the every lenth of audio files, the Convolution finished with every steps, which could be seen as the remainder of division is 0.
 
#### Setup Needed
- `setup.py`
- `requirement.txt`

#### Model Train
- `train.py`




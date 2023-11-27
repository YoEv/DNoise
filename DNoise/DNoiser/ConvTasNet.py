import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBLSTM(nn.Module):
    def __init__(self, dim, layers=2):
        super().__init__()
        self.layers = layers
        self.embedding = nn.Embedding(dim, dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=2), num_layers=layers)
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=layers, bidirectional=True)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        
        # Transformer Encoder
        x = x.permute(1, 0, 2)  # Adjust shape for Transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Restore original shape
        
        # BLSTM
        x, hidden = self.lstm(x, hidden)
        
        # Linear layer
        x = self.linear(x)
        return x, hidden
    
    def rescale_conv(self, conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference)**0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    def rescale_module(self, module, reference):
        for sub in module.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                self.rescale_conv(sub, reference)

    def rescale(self, reference):
        # Rescale Convolutional Layers in Transformer Encoder
        self.rescale_module(self.transformer_encoder, reference)

        # Rescale Convolutional Layers in LSTM
        self.rescale_module(self.lstm, reference)


# ConvTasNet with TransformerBLSTM
class ConvTasNetWithTransformerBLSTM(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000,
                 dim=256,  # Assuming dim value for TransformerBLSTM
                 layers=2):  # Assuming layers value for TransformerBLSTM

        super(ConvTasNetWithTransformerBLSTM, self).__init__()

        # TransformerBLSTM initialization
        self.transformer = TransformerBLSTM(dim=dim, layers=layers)

        # ConvTasNet initialization parameters
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        # Encoder and Decoder initialization similar to Demucs
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = nn.LSTM(dim, dim, bidirectional=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the ConvTasNet model so that
        there is no time steps left over in convolutions, e.g., for all
        layers, size of the input - kernel_size % stride = 0.
    
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)
    
    @property       
    def total_stride(self):
        """
        Calculate the total stride of the ConvTasNet model.
        """
        return self.stride ** self.depth // self.resample
    
            # Valid length calculation for ConvTasNet


    def forward(self, mix):
        # TransformerBLSTM forward pass
        x_blstm, _ = self.transformer(mix) ##########################################################################

        # ConvTasNet forward pass similar to Demucs
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x




class ConvTasNet(nn.Module):
    # 其他方法和属性...




    
    def separate_frame(self, frame):
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * self.resample
        x = frame.unsqueeze(0)  # 添加一个维度，表示batch_size为1
        for idx, encode in enumerate(self.encoder):
            stride //= self.stride
            length = x.shape[2]
            if idx == self.depth - 1:
                # This is slightly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - self.kernel_size) // self.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - self.kernel_size - self.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = torch.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = self.lstm(x, self.lstm_state)
        x = x.permute(1, 2, 0)

        extra = None
        for idx, decode in enumerate(self.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -self.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -self.stride:]
            else:
                extra[..., :self.stride] += next_state[-1]
            x = x[..., :-self.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :self.stride] += prev
            if idx != self.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]

import torch
import numpy as np
import yaml, logging, math
from random import shuffle


with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class SpeechSplit(torch.nn.Module):
    def __init__(self):
        super(SpeechSplit, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Pitch_Quantinizer'] = Quantinizer(hp_Dict['Sound']['Quantinized_Pitch_Dim'])

        self.layer_Dict['Rhyme_Encoder'] = Encoder(
            input_channels= hp_Dict['Sound']['Mel_Dim'],
            conv_channels= hp_Dict['Encoder']['Rhyme']['Conv']['Channels'],
            conv_kernel_sizes= hp_Dict['Encoder']['Rhyme']['Conv']['Kernel_Sizes'],
            norm_groups= hp_Dict['Encoder']['Rhyme']['Norm_Grous'],
            lstm_stacks= hp_Dict['Encoder']['Rhyme']['LSTM']['Stacks'],
            lstm_size= hp_Dict['Encoder']['Rhyme']['LSTM']['Sizes'],
            frequency= hp_Dict['Encoder']['Rhyme']['Frequency'],
            use_random_resampling= False
            )
        self.layer_Dict['Content_Encoder'] = Encoder(
            input_channels= hp_Dict['Sound']['Mel_Dim'],
            conv_channels= hp_Dict['Encoder']['Content']['Conv']['Channels'],
            conv_kernel_sizes= hp_Dict['Encoder']['Content']['Conv']['Kernel_Sizes'],
            norm_groups= hp_Dict['Encoder']['Content']['Norm_Grous'],
            lstm_stacks= hp_Dict['Encoder']['Content']['LSTM']['Stacks'],
            lstm_size= hp_Dict['Encoder']['Content']['LSTM']['Sizes'],
            frequency= hp_Dict['Encoder']['Content']['Frequency'],
            use_random_resampling= True
            )
        self.layer_Dict['Pitch_Encoder'] = Encoder(
            input_channels= hp_Dict['Sound']['Quantinized_Pitch_Dim'],
            conv_channels= hp_Dict['Encoder']['Pitch']['Conv']['Channels'],
            conv_kernel_sizes= hp_Dict['Encoder']['Pitch']['Conv']['Kernel_Sizes'],
            norm_groups= hp_Dict['Encoder']['Pitch']['Norm_Grous'],
            lstm_stacks= hp_Dict['Encoder']['Pitch']['LSTM']['Stacks'],
            lstm_size= hp_Dict['Encoder']['Pitch']['LSTM']['Sizes'],
            frequency= hp_Dict['Encoder']['Pitch']['Frequency'],
            use_random_resampling= True
            )

        self.layer_Dict['Decoder'] = Decoder(
            input_channels= \
                hp_Dict['Encoder']['Rhyme']['LSTM']['Sizes'] * 2 + \
                hp_Dict['Encoder']['Content']['LSTM']['Sizes'] * 2 + \
                hp_Dict['Encoder']['Pitch']['LSTM']['Sizes'] * 2 + \
                hp_Dict['Num_Speakers'], #one-hot
            lstm_stacks= hp_Dict['Decoder']['LSTM']['Stacks'],
            lstm_size= hp_Dict['Decoder']['LSTM']['Sizes'],
            )

    def forward(self, rhymes, contents, pitches, speakers, random_resampling_factors= None):
        assert contents.size(2) == pitches.size(1)

        pitches = self.layer_Dict['Pitch_Quantinizer'](pitches).transpose(2, 1)
        speakers = torch.nn.functional.one_hot(speakers, hp_Dict['Num_Speakers']).float()
        
        rhymes = self.layer_Dict['Rhyme_Encoder'](rhymes)
        contents = self.layer_Dict['Content_Encoder'](contents, random_resampling_factors)
        pitches = self.layer_Dict['Pitch_Encoder'](pitches, random_resampling_factors)

        upsamples = torch.cat([rhymes, contents, pitches, speakers.unsqueeze(2).expand(-1, -1, rhymes.size(2))], dim= 1)
        converted_Mels = self.layer_Dict['Decoder'](upsamples)

        return converted_Mels
    

class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        conv_channels,
        conv_kernel_sizes,
        norm_groups,        
        lstm_stacks,
        lstm_size,
        frequency,
        use_random_resampling= False
        ):
        super(Encoder, self).__init__()

        self.num_Conv = len(conv_channels)
        self.frequency = frequency
        self.use_random_resampling = use_random_resampling
        self.layer_Dict = torch.nn.ModuleDict()
        
        previous_Channels = input_channels
        for index, (channels, kernel_Size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
            self.layer_Dict['Conv_{}'.format(index)] = torch.nn.Sequential()
            self.layer_Dict['Conv_{}'.format(index)].add_module('Conv', torch.nn.Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                bias= True
                ))
            self.layer_Dict['Conv_{}'.format(index)].add_module('GroupNorm', torch.nn.GroupNorm(
                num_groups= norm_groups,
                num_channels= channels
                ))
            # self.layer_Dict['Conv'].add_module('ReLU_{}'.format(index), torch.nn.ReLU(
            #     inplace= True
            #     ))
            previous_Channels = channels

        if use_random_resampling:
            self.layer_Dict['Random_Resampling'] = Random_Resampling()

        self.layer_Dict['BiLSTM'] = torch.nn.LSTM(
            input_size= previous_Channels,
            hidden_size= lstm_size,
            num_layers= lstm_stacks,
            bias= True,
            batch_first= True,
            bidirectional= True
            )            

    def forward(self, x, factors= None):
        for index in range(self.num_Conv):
            x = self.layer_Dict['Conv_{}'.format(index)](x)            
            if self.use_random_resampling:
                x = self.layer_Dict['Random_Resampling'](x, factors if factors is None else factors[index])
        x = self.layer_Dict['BiLSTM'](x.transpose(2,1))[0].transpose(2,1)
        x_Forward, x_Backward = x.split(x.size(1) // 2, dim= 1) # [Batch, LSTM_dim, Time] * 2        
        x = torch.cat([
            x_Forward[:, :,self.frequency-1::self.frequency],
            x_Backward[:, :,::self.frequency]
            ], dim= 1)  # [Batch, LSTM_dim * 2, Time // Frequency]

        return x.repeat_interleave(self.frequency, dim= 2)

class Random_Resampling(torch.nn.Module):
    def forward(self, x, factors):
        if factors is None:
            return x

        resamples = []
        for length, factor in factors:
            samples, x = x[:, :, :length], x[:, :, length:]
            resamples.append(
                torch.nn.functional.interpolate(input= samples, size= factor, mode= 'linear', align_corners= True)
                )
        resamples = torch.cat(resamples, dim= 2)

        return resamples

class Decoder(torch.nn.Module):
    def __init__(self, input_channels, lstm_stacks, lstm_size):
        super(Decoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        
        self.layer_Dict['BiLSTM'] = torch.nn.LSTM(
            input_size= input_channels,
            hidden_size= lstm_size,
            num_layers= lstm_stacks,
            bias= True,
            batch_first= True,
            bidirectional= True
            )
        self.layer_Dict['Conv1x1'] = torch.nn.Conv1d(
            in_channels= lstm_size * 2,
            out_channels= hp_Dict['Sound']['Mel_Dim'],
            kernel_size= 1,
            bias= True            
            )      

    def forward(self, x):
        x = self.layer_Dict['BiLSTM'](x.transpose(2, 1))[0].transpose(2, 1)
        x = self.layer_Dict['Conv1x1'](x)

        return x

class Quantinizer(torch.nn.Module):
    def __init__(self, size):
        super(Quantinizer, self).__init__()
        self.size = size

    def forward(self, x):
        x = (x * self.size * 0.999).long()

        return torch.nn.functional.one_hot(x, num_classes= self.size).float()


if __name__ == "__main__":
    pass   

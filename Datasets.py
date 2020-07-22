import torch
import numpy as np
import yaml, pickle, os, math
from random import shuffle
#from Pattern_Generator import Pattern_Generate
from Pattern_Generator_Replication import Pattern_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

def random_resmpling_factors(length):
    factors = []
    left_Length = length
    left_New_Length = length
    while True:
        piece = np.random.randint(
            low= hp_Dict['Encoder']['Random_Resample']['Min_Length'],
            high= hp_Dict['Encoder']['Random_Resample']['Max_Length'] + 1
            )
        factor = int(piece * np.random.uniform(
            low= hp_Dict['Encoder']['Random_Resample']['Min_Factor'] if left_Length >= left_New_Length else 1.0,
            high= hp_Dict['Encoder']['Random_Resample']['Max_Factor'] if left_Length <= left_New_Length else 1.0
            ))
        factors.append((piece, factor))

        left_Length -= piece
        left_New_Length -= factor

        if left_Length < hp_Dict['Encoder']['Random_Resample']['Min_Length']:
            factors[0] = (factors[0][0] + left_Length, factors[0][1] + left_New_Length)
            break
        elif left_Length <= hp_Dict['Encoder']['Random_Resample']['Max_Length']:
            factors.append((left_Length, left_New_Length))
            break

    if any([x < 1 for factor in factors for x in factor]):
        return random_resmpling_factors(length) # To avoid the case that there is 0 or negative.

    shuffle(factors)
    return factors

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Train_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], hp_Dict['Train']['Train_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            # if metadata_Dict['Mel_Length_Dict'][x] > hp_Dict['Train']['Train_Pattern']['Pattern_Length']
            ] * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Speaker_Index'], pattern_Dict['Mel'], pattern_Dict['Pitch']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Dev_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dev_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], hp_Dict['Train']['Eval_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            ]
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Speaker_Index'], pattern_Dict['Mel'], pattern_Dict['Pitch']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path= 'Wav_Path_for_Inference_Replication.txt'):
        super(Inference_Dataset, self).__init__()

        self.pattern_List = [
            line.strip().split('\t')
            for line in open(pattern_path, 'r').readlines()[1:]
            ]
        
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        speaker_ID, rhythm_Label, rhythm_Path, content_Label, content_Path, pitch_Label, pitch_Path = self.pattern_List[idx]
        speaker_ID = int(speaker_ID)
        _, rhythm, _ = Pattern_Generate(rhythm_Path, 15)
        _, content, _ = Pattern_Generate(content_Path, 15)
        _, _, pitch = Pattern_Generate(pitch_Path)

        pattern = speaker_ID, rhythm, content, pitch, rhythm_Label, content_Label, pitch_Label

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)



class Collater:
    def __call__(self, batch):
        speakers, mels, pitches = zip(*[
            (speaker_ID, mel, pitch)
            for speaker_ID, mel, pitch in batch
            ])
        mels, pitches = self.Stack(mels, pitches, size= hp_Dict['Train']['Train_Pattern']['Pattern_Length'])
        factors = [
            random_resmpling_factors(mels.shape[1])
            for _ in range(len(hp_Dict['Encoder']['Content']['Conv']['Channels']))
            ]

        speakers = torch.LongTensor(speakers)   # [Batch]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        return speakers, mels, pitches, factors
    
    def Stack(self, mels, pitches, size= None):
        if size is None:
            max_Length = max([mel.shape[0] for mel in mels])

            mels = np.stack([
                np.pad(mel, [(0, max_Length - mel.shape[0]), (0, 0)], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
                for mel in mels
                ], axis= 0)
            pitches = np.stack([
                np.pad(pitch, (0, max_Length - pitch.shape[0]), constant_values= 0.0)
                for pitch in pitches
                ], axis= 0)
            return mels, pitches
        
        mel_List = []
        pitch_List = []
        for mel, pitch in zip(mels, pitches):
            if mel.shape[0] > size:
                offset = np.random.randint(0, mel.shape[0] - size)
                mel = mel[offset:offset + size]
                pitch = pitch[offset:offset + size]
            else:
                pad = size - mel.shape[0]
                mel = np.pad(
                    mel,
                    [[int(np.floor(pad / 2)), int(np.ceil(pad / 2))], [0, 0]],
                    mode= 'reflect'
                    )
                pitch = np.pad(
                    pitch,
                    [int(np.floor(pad / 2)), int(np.ceil(pad / 2))],
                    mode= 'reflect'
                    )                
            mel_List.append(mel)
            pitch_List.append(pitch)

        return np.stack(mel_List, axis= 0), np.stack(pitch_List, axis= 0)

class Inference_Collater:
    def __init__(self):
        frequency_LCM = \
            (hp_Dict['Encoder']['Rhythm']['Frequency'] * hp_Dict['Encoder']['Content']['Frequency']) // \
            math.gcd(hp_Dict['Encoder']['Rhythm']['Frequency'], hp_Dict['Encoder']['Content']['Frequency'])
        frequency_LCM = \
            (frequency_LCM * hp_Dict['Encoder']['Pitch']['Frequency']) // \
            math.gcd(frequency_LCM, hp_Dict['Encoder']['Pitch']['Frequency'])
        self.freqency_LCM = frequency_LCM
        
    def __call__(self, batch):
        speakers, rhythms, contents, pitches, rhythm_Labels, content_Labels, pitch_Labels = zip(*[
            (speaker_ID, rhythm, content, pitch, rhythm_Label, content_Label, pitch_Label)
            for speaker_ID, rhythm, content, pitch, rhythm_Label, content_Label, pitch_Label in batch
            ])

        rhythms, contents, pitches, lengths = self.Stacks(rhythms, contents, pitches)

        speakers = torch.LongTensor(speakers)   # [Batch]
        rhythms = torch.FloatTensor(rhythms).transpose(2, 1)   # [Batch, Mel_dim, Time]
        contents = torch.FloatTensor(contents).transpose(2, 1)   # [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        return speakers, rhythms, contents, pitches, rhythm_Labels, content_Labels, pitch_Labels, lengths

    def Stacks(self, rhythms, contents, pitches):
        max_Length = max([rhythm.shape[0] for rhythm in rhythms])
        max_Length = math.ceil(max_Length / self.freqency_LCM) * self.freqency_LCM

        lengths = [rhythm.shape[0] for rhythm in rhythms]

        rhythms = np.stack([
            np.pad(rhythm, [(0, max_Length - rhythm.shape[0]), (0, 0)], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
            for rhythm in rhythms
            ], axis= 0)
        
        contents = [
            torch.nn.functional.interpolate(
                input= torch.FloatTensor(content).transpose(0, 1).unsqueeze(0),
                size= length,
                mode= 'linear',
                align_corners= True
                ).squeeze(0).transpose(0, 1).numpy()
            for content, length in zip(contents, lengths)
            ]
        contents = np.stack([
            np.pad(content, [(0, max_Length - content.shape[0]), (0, 0)], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
            for content in contents
            ], axis= 0)
        
        pitches = [
            torch.nn.functional.interpolate(
                input= torch.FloatTensor(pitch).unsqueeze(0).unsqueeze(0),
                size= length,
                mode= 'linear',
                align_corners= True
                ).squeeze(0).squeeze(0).numpy()
            for pitch, length in zip(pitches, lengths)
            ]
        pitches = np.stack([
            np.pad(pitch, (0, max_Length - pitch.shape[0]))
            for pitch in pitches
            ], axis= 0)
        
        return rhythms, contents, pitches, lengths


if __name__ == "__main__":    
    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= Train_Dataset(),
    #     shuffle= True,
    #     collate_fn= Collater(),
    #     batch_size= hp_Dict['Train']['Batch_Size'],
    #     num_workers= hp_Dict['Train']['Num_Workers'],
    #     pin_memory= True
    #     )

    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= Dev_Dataset(),
    #     shuffle= True,
    #     collate_fn= Collater(),
    #     batch_size= hp_Dict['Train']['Batch_Size'],
    #     num_workers= hp_Dict['Train']['Num_Workers'],
    #     pin_memory= True
    #     )

    # import time
    # for x in dataLoader:
    #     speakers, mels, pitches, factors = x
    #     print(speakers.shape)
    #     print(mels.shape)
    #     print(pitches.shape)
    #     print(factors)
    #     time.sleep(2.0)

    dataLoader = torch.utils.data.DataLoader(
        dataset= Inference_Dataset(),
        shuffle= False,
        collate_fn= Inference_Collater(),
        batch_size= hp_Dict['Train']['Batch_Size'],
        num_workers= hp_Dict['Train']['Num_Workers'],
        pin_memory= True
        )

    import time
    for x in dataLoader:
        speakers, rhythms, contents, pitches, factors, rhythm_Labels, content_Labels, pitch_Labels, lengths = x
        print(speakers.shape)
        print(rhythms.shape)
        print(contents.shape)
        print(pitches.shape)
        print(factors)
        print(rhythm_Labels)
        print(content_Labels)
        print(pitch_Labels)
        print(lengths)
        time.sleep(2.0)








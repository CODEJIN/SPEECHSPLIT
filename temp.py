# import pickle
# import matplotlib.pyplot as plt
# from yin import pitch_calc
# from Audio import preemphasis
# import librosa
# import numpy as np

# # x = pickle.load(open('C:/Pattern/SS.Pattern/Train/VCTK/VCTK.P225.p225_006.PICKLE', 'rb'))
# sig = librosa.load("D:/Pattern/ENG/VCTK/wav48/p225/p225_001.wav", 16000)[0]
# sig = librosa.util.normalize(sig)
# sig = librosa.effects.trim(sig, 15)[0]
# print(sig)
# pitch_calc(sig, 16000, f0_min= 100, confidence_threshold= 0.6, gaussian_smoothing_sigma= 1.0)
# assert False
# plt.subplot(411)
# plt.plot(x['Audio'])
# plt.subplot(412)
# plt.imshow(x['Mel'].T, aspect='auto', origin='lower')
# plt.subplot(413)
# plt.plot(x['Pitch'])
# plt.subplot(414)
# plt.plot(pitch_calc(preemphasis(x['Audio']), 16000, 1024, 256, 100, 500, 0.85, 1.0))
# plt.show()

# print(np.max(x['Pitch']))


# import numpy as np
# import matplotlib.pyplot as plt
# import librosa
# from yin import pitch_calc

# sig = librosa.load("D:/Pattern/ENG/VCTK/wav48/p225/p225_001.wav", 16000)[0]
# a = pitch_calc(sig, 16000, confidence_threshold= 0.6, gaussian_smoothing_sigma= 0.0)
# b = (a - np.mean(a)) / np.std(a)
# c = (a - np.min(a)) / (np.max(a) - np.min(a))
# d = (b - np.min(b)) / (np.max(b) - np.min(b))

# print(c - d)

# plt.subplot(411)
# plt.plot(a)
# plt.subplot(412)
# plt.plot(b)
# plt.subplot(413)
# plt.plot(c)
# plt.subplot(414)
# plt.plot(d)
# plt.show()

# id_Dict = {'P243': 0, 'P240': 19}
# path_Dict = {
#     ('P243', 'Consistent'): './inference_wavs/p243_001.wav',
#     ('P243', 'Trained_Male_Inconsistent'): './inference_wavs/p232_005.wav',
#     ('P243', 'Trained_Female_Inconsistent'): './inference_wavs/p277_006.wav',
#     ('P243', 'Unseen_Male_Inconsistent'): './inference_wavs/p226_003.wav',
#     ('P243', 'Unseen_Female_Inconsistent'): './inference_wavs/p228_004.wav',
#     ('P240', 'Consistent'): './inference_wavs/p240_002.wav',
#     ('P240', 'Trained_Male_Inconsistent'): './inference_wavs/p232_005.wav',
#     ('P240', 'Trained_Female_Inconsistent'): './inference_wavs/p277_006.wav',
#     ('P240', 'Unseen_Male_Inconsistent'): './inference_wavs/p226_003.wav',
#     ('P240', 'Unseen_Female_Inconsistent'): './inference_wavs/p228_004.wav',
#     ('P226', 'Consistent'): './inference_wavs/p226_003.wav',
#     ('P226', 'Trained_Male_Inconsistent'): './inference_wavs/p232_005.wav',
#     ('P226', 'Trained_Female_Inconsistent'): './inference_wavs/p277_006.wav',
#     ('P226', 'Unseen_Male_Inconsistent'): './inference_wavs/p226_003.wav',
#     ('P226', 'Unseen_Female_Inconsistent'): './inference_wavs/p228_004.wav',
#     ('P228', 'Consistent'): './inference_wavs/p228_004.wav',
#     ('P228', 'Trained_Male_Inconsistent'): './inference_wavs/p232_005.wav',
#     ('P228', 'Trained_Female_Inconsistent'): './inference_wavs/p277_006.wav',
#     ('P228', 'Unseen_Male_Inconsistent'): './inference_wavs/p226_003.wav',
#     ('P228', 'Unseen_Female_Inconsistent'): './inference_wavs/p228_004.wav',
#     }

# open('x.txt', 'w').write('\n'.join([
#     '\t'.join([str(index), r, path_Dict[speaker, r], c, path_Dict[speaker, c], p, path_Dict[speaker, p]])
#     for speaker, index in id_Dict.items()
#     for r in ['Consistent', 'Trained_Male_Inconsistent', 'Trained_Female_Inconsistent', 'Unseen_Male_Inconsistent', 'Unseen_Female_Inconsistent']
#     for c in ['Consistent', 'Trained_Male_Inconsistent', 'Trained_Female_Inconsistent', 'Unseen_Male_Inconsistent', 'Unseen_Female_Inconsistent']
#     for p in ['Consistent', 'Trained_Male_Inconsistent', 'Trained_Female_Inconsistent', 'Unseen_Male_Inconsistent', 'Unseen_Female_Inconsistent']    
#     ]))

# import pickle            
# x = pickle.load(open("C:\Pattern\SS.Replication.Pattern\Train\METADATA.PICKLE", 'rb'))
# for file in x['File_List']:
#     if x['Mel_Length_Dict'][file] - x['Pitch_Length_Dict'][file] != 4:
#         print(x['Mel_Length_Dict'][file], x['Pitch_Length_Dict'][file])

# import pickle, os
# import numpy as np
# from yin import pitch_calc
# from Audio import melspectrogram




# for root, _, files in os.walk('C:/Pattern/SS.Replication.Pattern'):
#     for file in files:
#         path = os.path.join(root, file).replace('\\', '/')
#         pattern = pickle.load(open(path, 'rb'))
#         difference = (pattern['Mel'].shape[0] - pattern['Pitch'].shape[0])
#         pattern['Pitch'] = np.pad(pattern['Pitch'], (np.floor(difference / 2).astype(np.int64), 0), constant_values= pattern['Pitch'][0])
#         pattern['Pitch'] = np.pad(pattern['Pitch'], (0, np.ceil(difference / 2).astype(np.int64)), constant_values= pattern['Pitch'][-1])

#         with open(path, 'wb') as f:
#             pickle.dump(pattern, f, protocol=4)

import numpy as np
import matplotlib.pyplot as plt
import librosa
from yin import pitch_calc
from Audio import melspectrogram
import torch

sig = librosa.load("D:/Pattern/ENG/VCTK/wav48/p225/p225_001.wav", 16000)[0]
sig = librosa.effects.trim(sig, 15, frame_length= 32, hop_length= 16)[0]
sig = librosa.util.normalize(sig)

a = pitch_calc(sig, 16000, confidence_threshold= 0.6, gaussian_smoothing_sigma= 0.0)
b = melspectrogram(sig, 1025, 256, 1024, 80, 16000, 4)
c = torch.nn.functional.interpolate(input= torch.FloatTensor(a).unsqueeze(0).unsqueeze(0), size= 400, mode= 'linear', align_corners= True).squeeze(0).squeeze(0).numpy()
d = torch.nn.functional.interpolate(input= torch.FloatTensor(b).unsqueeze(0), size= 400, mode= 'linear', align_corners= True).squeeze(0).numpy()

print(a.shape, b.shape, c.shape, d.shape)
print(torch.nn.functional.interpolate(input= torch.FloatTensor(b).unsqueeze(0), size= 400, mode= 'linear', align_corners= True).shape)

plt.subplot(411)
plt.plot(a)
plt.margins(x=0)
plt.subplot(412)
plt.imshow(b, aspect='auto', origin='lower')
plt.subplot(413)
plt.plot(c)
plt.margins(x=0)
plt.subplot(414)
plt.imshow(d, aspect='auto', origin='lower')
plt.show()
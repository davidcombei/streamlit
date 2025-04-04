import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
audio_processor = AudioProcessor()


####################################
#### DEFINE THE DECODER ############
####################################
class ADDvisor(nn.Module):
    def __init__(self, wav2vec2_dim=1920, num_freq_bins=int((audio_processor.n_fft//2)+1), time_steps_w2v = (audio_processor.audio_length*50)-1 ,
                 time_steps_stft =int(audio_processor.audio_length * (audio_processor.sampling_rate / audio_processor.hop_length))+1, threshold = 0.3):
        super(ADDvisor, self).__init__()



        self.threshold = threshold
        self.num_freq_bins = num_freq_bins
        self.time_steps_w2v = time_steps_w2v
        self.time_steps_stft = time_steps_stft


        self.conv1 = nn.Conv1d(wav2vec2_dim, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(768, self.num_freq_bins, kernel_size=3, stride=1, padding=1)


        self.sigmoid = nn.Sigmoid()



    def forward(self, h_w2v):

        x = h_w2v.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        mask = torch.sigmoid(x)

        return mask



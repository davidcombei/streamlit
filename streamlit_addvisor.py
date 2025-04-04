import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from addvisor import ADDvisor
from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg, TorchScaler, thresh
import os
from collections import OrderedDict
from tqdm import tqdm
import io
from torch.utils.data import Dataset, DataLoader

st.set_page_config(layout="wide")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_processor = AudioProcessor()
model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)

checkpoint_path = r'C:\Users\david\PycharmProjects\David2\model\addvisor_MLAAD_log1pMAG_epoch_21_loss_0.0941.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
if any(k.startswith("module.") for k in checkpoint.keys()):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    checkpoint = new_state_dict
model.load_state_dict(checkpoint)

@st.cache_data
def plot_spectrogram(spec, title):
    fig, ax = plt.subplots()
    ax.imshow(np.log1p(spec), aspect='auto', origin='lower')
    ax.set_title(title)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

class AudioDataset(Dataset):
    def __init__(self, directory, audio_processor, device):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')][:50]
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform = self.audio_processor.load_audio(audio_path)[0]
        return waveform.to(self.device), audio_path

@st.cache_resource(show_spinner=True)
def run_addvisor_batched(dir_path):
    dataset = AudioDataset(directory=dir_path, audio_processor=audio_processor, device=device)
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)
    results = []
    for batch in tqdm(data_loader):
        waveforms, paths = batch
        feats = audio_processor.extract_features(waveforms)
        feats_mean = torch.mean(feats, dim=1)

        yhat1_logits, yhat1_probs = torch_log_reg(feats_mean)
        mask = model(feats)
        threshold = mask.mean()
        mask = (mask > threshold).float()
        print(mask)
        print(mask.max().item())
        _, magnitude, phase = audio_processor.compute_stft(waveforms)
        Tmax = mask.shape[1]
        #magnitude = magnitude.to(device)
        log_mag = torch.log1p(magnitude[:, :Tmax, :]).to(device)

        phase = phase[:, :Tmax, :].to(device)
        relevant_mask_stft = mask * log_mag#magnitude[:, :Tmax, :]
        relevant_mask_stft = torch.expm1(relevant_mask_stft)
        relevant_mask = relevant_mask_stft * torch.exp(1j * phase)
        print((mask * log_mag).sum().item())
        print(((1 - mask) * log_mag).sum().item())
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_feats = audio_processor.extract_features_istft(istft_relevant_mask)
        istft_feats_mean = torch.mean(istft_feats, dim=1)
        yhat2_logits, yhat2_probs = torch_log_reg(istft_feats_mean)



        for i in range(waveforms.size(0)):
            results.append({
                "filename": os.path.basename(paths[i]),
                "original_audio": waveforms[i].cpu().numpy(),
                "reconstructed_audio": istft_relevant_mask[i].detach().cpu().numpy(),
                "spectrogram_img": plot_spectrogram(magnitude[i].detach().cpu().numpy(), "Spectrogram"),
                "mask_img": plot_spectrogram(mask[i].cpu().detach().numpy(), "Mask"),
                "masked_spectrogram_img": plot_spectrogram(relevant_mask_stft[i].cpu().detach().numpy(), "Spectrogram x Mask "),
                "pred_original": yhat1_probs[i].cpu().detach().numpy(),
                "pred_reconstructed": yhat2_probs[i].cpu().detach().numpy()
            })
    return results

#DIR_PATH = "C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav"
#DIR_PATH = 'C:\Machine_Learning_Data\con_wav'
DIR_PATH = r"C:\Machine_Learning_Data\Deepfake_datasets\mlaad_v5\fake\en\MatchaTTS"
results = run_addvisor_batched(DIR_PATH)

st.title("quality visualization of explainability")

for item in results:
    st.subheader(item["filename"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**original audio**")
        st.audio(item["original_audio"], format="audio/wav", sample_rate=16000)
    with col2:
        st.markdown("**reconstructed audio**")
        st.audio(item["reconstructed_audio"], format="audio/wav", sample_rate=16000)

    st.image(item["spectrogram_img"], caption="spectrogram", use_column_width=True)
    st.image(item["mask_img"], caption="mask", use_column_width=True)
    st.image(item["masked_spectrogram_img"], caption="spectrogram x mask", use_column_width=True)

    st.markdown("**predictions**")
    st.write("on original audio: ", item["pred_original"])
    st.write("on reconstructed audio: ", item["pred_reconstructed"])
    st.markdown("---")

import json
import os
import tempfile
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn

mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import sys
sys.path.append('.')
sys.path.append('hifi-gan/')
sys.path.append('speaker_encoder/')

import params
from model import DiffVC
from env import AttrDict
from models import Generator as HiFiGAN
from encoder import inference as spk_encoder

app = FastAPI(title="DiffVC Voice Conversion Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ──────────────────────────────────────────────────────────────────

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256,
                             win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i + 2 * w + 1])
        y[i] = min(x[i + w + 1], med)
    return y


def mel_spectral_subtraction(mel_synth, mel_source,
                              spectral_floor=0.02,
                              silence_window=5,
                              smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min, i_min = 100000.0, 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i + silence_window]))
        if energy_cur < energy_min:
            i_min, energy_min = i, energy_cur
    estimated_noise_energy = np.min(
        np.exp(2.0 * mel_synth[:, i_min:i_min + silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(
            estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(
            signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


def to_device(tensor):
    return tensor.cuda() if use_gpu else tensor


# ── model loading (on startup) ────────────────────────────────────────────────

generator = None
hifigan = None


@app.on_event("startup")
def load_models():
    global generator, hifigan

    # Voice conversion model
    vc_path = 'checkpts/vc/vc_libritts_wodyn.pt'
    generator = DiffVC(
        params.n_mels, params.channels, params.filters, params.heads,
        params.layers, params.kernel, params.dropout, params.window_size,
        params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
        params.beta_min, params.beta_max)
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator.eval()

    # HiFi-GAN vocoder
    hfg_path = 'checkpts/vocoder/'
    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))
    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(
            torch.load(hfg_path + 'generator')['generator'])
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(
            torch.load(hfg_path + 'generator', map_location='cpu')['generator'])
    hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()
    hifigan = hifigan_universal

    # Speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')
    device = "cuda" if use_gpu else "cpu"
    spk_encoder.load_model(enc_model_fpath, device=device)

    print(f"✅ Models loaded | GPU: {use_gpu} | Params: {generator.nparams:,}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open("./webapp/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": use_gpu,
        "models_loaded": generator is not None,
    }


@app.post("/convert")
async def convert_voice(
    source: UploadFile = File(..., description="Source speech WAV file"),
    target: UploadFile = File(..., description="Target speaker reference WAV file"),
    n_timesteps: int = 30,
):
    """
    Convert the voice in **source** to sound like the speaker in **target**.

    Returns a WAV file with the converted speech.
    """
    if generator is None or hifigan is None:
        raise HTTPException(503, "Models not loaded yet, please retry.")

    # Save uploads to temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sf:
        sf.write(await source.read())
        src_path = sf.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(await target.read())
        tgt_path = tf.name

    try:
        # Mel spectrograms
        mel_source = to_device(
            torch.from_numpy(get_mel(src_path)).float().unsqueeze(0))
        mel_source_lengths = to_device(
            torch.LongTensor([mel_source.shape[-1]]))

        mel_target = to_device(
            torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0))
        mel_target_lengths = to_device(
            torch.LongTensor([mel_target.shape[-1]]))

        embed_target = to_device(
            torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0))

        # Voice conversion
        with torch.no_grad():
            _, mel_ = generator.forward(
                mel_source, mel_source_lengths,
                mel_target, mel_target_lengths,
                embed_target,
                n_timesteps=n_timesteps,
                mode='ml',
            )
            mel_synth_np = mel_.cpu().squeeze().numpy()
            mel_source_np = mel_source.cpu().squeeze().numpy()

            mel_denoised = mel_spectral_subtraction(
                mel_synth_np, mel_source_np, smoothing_window=1)
            mel_tensor = to_device(
                torch.from_numpy(mel_denoised).float().unsqueeze(0))

            audio = hifigan.forward(mel_tensor).cpu().squeeze().clamp(-1, 1)

        # Write output WAV
        out_fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        from scipy.io.wavfile import write as wav_write
        wav_write(out_path, 22050, audio.numpy())

        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=None,          # keep file alive for response
        )

    finally:
        os.unlink(src_path)
        os.unlink(tgt_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
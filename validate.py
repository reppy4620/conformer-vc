import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from models import ConformerVC
from hifi_gan import load_hifi_gan
from transform import TacotronSTFT

SR = 24000


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--hifi_gan', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./DATA')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    config = OmegaConf.load(f'{args.model_dir}/config.yaml')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = f'{args.model_dir}/latest.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ConformerVC(config.model)
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded {checkpoint["iteration"]} Iteration Model')
    hifi_gan = load_hifi_gan(args.hifi_gan)
    model, hifi_gan = model.eval().to(device), hifi_gan.eval().to(device)

    to_mel = TacotronSTFT()

    def infer(mel, length, pitch, energy):
        mel = mel.transpose(-1, -2).unsqueeze(0).to(device)
        pitch = pitch.transpose(-1, -2).unsqueeze(0).to(device)
        energy = energy.transpose(-1, -2).unsqueeze(0).to(device)
        length = torch.LongTensor([length]).to(device)
        with torch.no_grad():
            mel, pred_pitch = model.infer(mel, length, pitch, energy)
            wav = hifi_gan(mel)
            mel, wav = mel.cpu(), wav.squeeze(1).cpu()
            pred_pitch = pred_pitch.squeeze().cpu()
        return mel, wav, pred_pitch

    def save_wav(wav, path):
        torchaudio.save(
            str(path),
            wav,
            24000,
            encoding='PCM_S',
            bits_per_sample=16
        )

    def save_mel_three_attn(tgt, gen, gen2, path):
        plt.figure(figsize=(10, 7))
        plt.subplot(311)
        plt.gca().title.set_text('GT')
        plt.imshow(tgt, aspect='auto', origin='lower')
        plt.subplot(312)
        plt.gca().title.set_text('GEN')
        plt.imshow(gen, aspect='auto', origin='lower')
        plt.subplot(313)
        plt.gca().title.set_text('GEN(vocoder)')
        plt.imshow(gen2, aspect='auto', origin='lower')
        plt.savefig(path)
        plt.close()

    def save_f0(tgt, pred, path):
        plt.figure(figsize=(10, 7))
        plt.plot(tgt, label='GT')
        plt.plot(pred, label='Pred')
        plt.legend()
        plt.grid()
        plt.savefig(path)
        plt.close()

    fns = list(sorted(list(Path(args.data_dir).glob('*.pt'))))

    for fn in tqdm(fns, total=len(fns)):
        (
            src_wav,
            tgt_wav,
            src_mel,
            tgt_mel,
            src_length,
            tgt_length,
            src_pitch,
            tgt_pitch,
            src_energy,
            tgt_energy,
            _
        ) = torch.load(fn)
        mel_gen, wav_gen, pred_pitch = infer(src_mel, src_length, src_pitch, src_energy)
        mel_gen2, _ = to_mel(wav_gen)

        d = output_dir / os.path.splitext(fn.name)[0]
        d.mkdir(exist_ok=True)

        save_wav(src_wav, d / 'src.wav')
        save_wav(tgt_wav, d / 'tgt.wav')
        save_wav(wav_gen, d / 'gen.wav')

        save_mel_three_attn(
            tgt_mel.squeeze().transpose(0, 1),
            mel_gen.squeeze(),
            mel_gen2.squeeze(),
            d / 'comp.png'
        )

        save_f0(tgt_pitch, pred_pitch, d / 'f0.png')


if __name__ == '__main__':
    main()

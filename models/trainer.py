import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data import VCDataset, collate_fn
from .model import ConformerVC
from .discriminator import Discriminator
from .lr_scheduler import NoamLR
from .utils import rand_slice
from .loss import d_loss, g_loss, feature_map_loss
from utils import seed_everything, Tracker


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        config = OmegaConf.load(self.config_path)

        accelerator = Accelerator(fp16=config.train.fp16)

        seed_everything(config.seed)

        output_dir = Path(config.model_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(config, output_dir / 'config.yaml')

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=f'{str(output_dir)}/logs')
        else:
            writer = None

        train_data, valid_data = self.prepare_data(config.data)
        train_dataset = VCDataset(train_data)
        valid_dataset = VCDataset(valid_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.train.batch_size,
            num_workers=8,
            collate_fn=collate_fn
        )

        model_g = ConformerVC(config.model)
        model_d = Discriminator(in_channels=config.model.n_mel, **config.model.discriminator)
        optimizer_g = optim.AdamW(model_g.parameters(), eps=1e-9, **config.optimizer)
        optimizer_d = optim.AdamW(model_d.parameters(), eps=1e-9, **config.optimizer)

        epochs = self.load(config, model_g, model_d, optimizer_g, optimizer_g)

        model_g, model_d, optimizer_g, optimizer_d, train_loader, valid_loader = accelerator.prepare(
            model_g, model_d, optimizer_g, optimizer_d, train_loader, valid_loader
        )
        scheduler_g = NoamLR(optimizer_g, channels=config.model.encoder.channels, last_epoch=epochs * len(train_loader) - 1)
        scheduler_d = NoamLR(optimizer_d, channels=config.model.discriminator.channels, last_epoch=epochs * len(train_loader) - 1)

        for epoch in range(epochs, config.train.num_epochs):
            self.train_step(
                config,
                epoch,
                [model_g, model_d],
                [optimizer_g, optimizer_d],
                [scheduler_g, scheduler_d],
                train_loader,
                writer,
                accelerator
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.valid_step(epoch, model_g, valid_loader, writer)
                if (epoch + 1) % config.train.save_interval == 0:
                    self.save(
                        output_dir / 'latest.ckpt',
                        epoch,
                        (epoch+1)*len(train_loader),
                        accelerator.unwrap_model(model_g),
                        accelerator.unwrap_model(model_d),
                        optimizer_g,
                        optimizer_d
                    )
        if accelerator.is_main_process:
            writer.close()

    def train_step(self, config, epoch, models, optimizers, schedulers, loader, writer, accelerator):
        model_g, model_d = models
        optimizer_g, optimizer_d = optimizers
        scheduler_g, scheduler_d = schedulers
        model_g.train()
        model_d.train()

        tracker = Tracker()
        bar = tqdm(desc=f'Epoch: {epoch + 1}', total=len(loader), disable=not accelerator.is_main_process)
        for i, batch in enumerate(loader):
            (
                src_mel,
                tgt_mel,
                src_length,
                tgt_length,
                tgt_duration,
                src_pitch,
                tgt_pitch,
                src_energy,
                tgt_energy,
                path
            ) = batch

            x, x_post, (dur_pred, pitch_pred, energy_pred) = model_g(
                src_mel, src_length, tgt_length, src_pitch, tgt_pitch, src_energy, tgt_energy, path
            )

            # D
            b, e = rand_slice(tgt_length, segment_length=config.model.segment_length)
            pred_real, _ = model_d(tgt_mel[:, :, b:e])
            pred_fake, _ = model_d(x_post[:, :, b:e].detach())
            loss_d = d_loss(pred_real, pred_fake)
            optimizer_d.zero_grad()
            accelerator.backward(loss_d)
            accelerator.clip_grad_norm_(model_d.parameters(), 5)
            optimizer_d.step()
            scheduler_d.step()

            # G
            pred_real, fm_real = model_d(tgt_mel[:, :, b:e])
            pred_fake, fm_fake = model_d(x_post[:, :, b:e])
            loss_gan = g_loss(pred_fake)
            loss_fm = feature_map_loss(fm_real, fm_fake)
            loss_recon = F.l1_loss(x, tgt_mel)
            loss_post_recon = F.l1_loss(x_post, tgt_mel)
            loss_duration = F.mse_loss(dur_pred, tgt_duration.to(x.dtype))
            loss_pitch = F.mse_loss(pitch_pred, tgt_pitch.to(x.dtype))
            loss_energy = F.mse_loss(energy_pred, tgt_energy.to(x.dtype))
            loss_g = loss_gan + loss_fm + loss_recon + loss_post_recon + loss_duration + loss_pitch + loss_energy
            optimizer_g.zero_grad()
            accelerator.backward(loss_g)
            accelerator.clip_grad_norm_(model_g.parameters(), 5)
            optimizer_g.step()
            scheduler_g.step()

            tracker.update(
                loss_d=loss_d.item(),
                loss_g=loss_g.item(),
                recon=loss_recon.item(),
                post_recon=loss_post_recon.item(),
                duration=loss_duration.item(),
                pitch=loss_pitch.item(),
                energy=loss_energy.item()
            )

            bar.update()
            bar.set_postfix_str(f'Loss_g: {loss_g:.6f}, Loss_d: {loss_d: .6f}')
        bar.set_postfix_str(f'Mean Loss_g: {tracker.loss_g.mean():.6f}, Mean Loss_d: {tracker.loss_d.mean():.6f}')
        if accelerator.is_main_process:
            self.write_losses(epoch, writer, tracker, mode='train')
        bar.close()

    def valid_step(self, epoch, model_g, loader, writer):
        model_g.eval()
        tracker = Tracker()
        with torch.no_grad():
            for batch in loader:
                (
                    src_mel,
                    tgt_mel,
                    src_length,
                    tgt_length,
                    tgt_duration,
                    src_pitch,
                    tgt_pitch,
                    src_energy,
                    tgt_energy,
                    path
                ) = batch

                x, x_post, (dur_pred, pitch_pred, energy_pred) = model_g(
                    src_mel, src_length, tgt_length, src_pitch, tgt_pitch, src_energy, tgt_energy, path
                )
                loss_recon = F.l1_loss(x, tgt_mel)
                loss_post_recon = F.l1_loss(x_post, tgt_mel)
                loss_duration = F.mse_loss(dur_pred, tgt_duration.to(x.dtype))
                loss_pitch = F.mse_loss(pitch_pred, tgt_pitch.to(x.dtype))
                loss_energy = F.mse_loss(energy_pred, tgt_energy.to(x.dtype))
                loss_g = loss_recon + loss_post_recon + loss_duration + loss_pitch + loss_energy

                tracker.update(
                    loss_g=loss_g.item(),
                    recon=loss_recon.item(),
                    post_recon=loss_post_recon.item(),
                    duration=loss_duration.item(),
                    pitch=loss_pitch.item(),
                    energy=loss_energy.item()
                )
        self.write_losses(epoch, writer, tracker, mode='valid')

    def prepare_data(self, config):
        data_dir = Path(config.data_dir)
        assert data_dir.exists()

        fns = list(sorted(list(data_dir.glob('*.pt'))))
        train_size = 500
        train = fns[:train_size]
        valid = fns[train_size:]
        return train, valid

    def load(self, config, model_g, model_d, optimizer_g, optimizer_d):
        if config.resume_checkpoint:
            checkpoint = torch.load(f'{config.model_dir}/latest.ckpt')
            epochs = checkpoint['epoch']
            iteration = checkpoint['iteration']
            model_g.load_state_dict(checkpoint['model_g'])
            model_d.load_state_dict(checkpoint['model_d'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f'Loaded {iteration}iter model and optimizer.')
            return epochs + 1
        else:
            return 0

    def save(self, save_path, epoch, iteration, model_g, model_d, optimizer_g, optimizer_d):
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'model_g': model_g.state_dict(),
            'model_d': model_d.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict()
        }, save_path)

    def write_losses(self, epoch, writer, tracker, mode='train'):
        for k, v in tracker.items():
            writer.add_scalar(f'{mode}/{k}', v.mean(), epoch)

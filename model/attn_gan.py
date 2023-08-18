import math
import torch
import numpy as np
import lightning as L
from torch import nn, Tensor
import torchaudio
import wandb
import torchaudio.transforms as T
from itertools import chain
from functools import reduce
from torchvision.utils import make_grid
from .generator import AttentionGuideGenerator
from .discriminator import PatchGAN
from .utils import get_criterion, init_weights, ImagePool
import librosa

FIX_W = 128


class MagnitudeAttentionGAN(L.LightningModule):
    """
    call A, B as two domain considering
    in term of speech: A is clean speech and B is noisy speech
    # this class work for magnitude only
    """

    def __init__(self, cfg, total_steps: int = 0, istft_params: dict = {}):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.gen_A2B = AttentionGuideGenerator(**self.hparams.cfg.generator)
        self.gen_B2A = AttentionGuideGenerator(**self.hparams.cfg.generator)

        # first adversarial loss
        self.disc_A = PatchGAN(**self.hparams.cfg.discriminator)
        self.disc_B = PatchGAN(**self.hparams.cfg.discriminator)

        # for second adversarial loss
        self.disc_A2 = PatchGAN(**self.hparams.cfg.discriminator)
        self.disc_B2 = PatchGAN(**self.hparams.cfg.discriminator)

        self.idt_loss = get_criterion(cfg.criterion.idt_loss)
        self.cycle_loss = get_criterion(cfg.criterion.cycle_loss)
        self.adv_loss = get_criterion(cfg.criterion.gan_loss)

        self.stft = T.Spectrogram(**istft_params, power=None)

        custom_src_audio_path = "/home/stud_vantuan/share_with_150/cache/helicopter_1h_30m/test/clean/5514-19192-0038.flac"
        wav, sr = torchaudio.load(custom_src_audio_path)
        spectrogram = self.stft(wav)
        magnitude = torch.abs(spectrogram)
        self.phase = torch.angle(spectrogram)
        self.mag_coms, self.max_size = self.cutting(magnitude)
        self.mag_coms = torch.stack(self.mag_coms, dim=0)
        self.sr = sr

        init_weights(self, **cfg.init_params)

        self.fake_A_pool = ImagePool(**cfg.image_pool)
        self.fake_B_pool = ImagePool(**cfg.image_pool)
        self.cycle_A_pool = ImagePool(**cfg.image_pool)
        self.cycle_B_pool = ImagePool(**cfg.image_pool)

    def cutting(self, img, fix_w: int = FIX_W):
        max_size = img.size(-1)
        l = []
        curr_idx = 0
        while img.size(-1) - curr_idx > fix_w:
            l.append(img[:, :, curr_idx : curr_idx + fix_w])
            curr_idx += fix_w
        remain = max_size - curr_idx
        if remain:
            add = fix_w - remain
            roll_img = torch.tile(img, (math.ceil(add / fix_w),))
            remain_tensor = img[:, :, curr_idx:]
            add_on_tensor = roll_img[:, :, :add]
            l.append(torch.cat([remain_tensor, add_on_tensor], dim=-1))
        return l, max_size

    def cal_adv_loss(self, predict, is_real):
        if is_real:
            label = torch.ones_like(predict).type_as(predict)
        else:
            label = torch.zeros_like(predict).type_as(predict)
        return self.adv_loss(predict, label)

    def cal_accuracy(self, predict: Tensor, threshold: float = 0.5):
        return (predict.sigmoid() >= threshold).sum() / reduce(
            lambda x, y: x * y, predict.size()
        )

    def configure_optimizers(self):
        if self.hparams.cfg.optimizer.name == "adam":
            optimizer_class = torch.optim.Adam
        elif self.hparams.cfg.optimizer.name == "adamw":
            optimizer_class = torch.optim.AdamW

        if self.hparams.cfg.scheduler.name == "one_cycle_lr":
            scheduler_class = torch.optim.lr_scheduler.OneCycleLR
        else:
            scheduler_class = None

        g_params = [self.gen_A2B.parameters(), self.gen_B2A.parameters()]
        d_params = [
            self.disc_A.parameters(),
            self.disc_A2.parameters(),
            self.disc_B.parameters(),
            self.disc_B2.parameters(),
        ]
        optim_g = optimizer_class(
            chain(*g_params), **self.hparams.cfg.optimizer.params_gen
        )
        optim_d = optimizer_class(
            chain(*d_params), **self.hparams.cfg.optimizer.params_disc
        )

        if scheduler_class != None:
            scheduler_g = {
                "scheduler": scheduler_class(
                    optim_g,
                    total_steps=self.hparams.total_steps,
                    **self.hparams.cfg.scheduler.params_gen
                ),
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            }

            scheduler_d = {
                "scheduler": scheduler_class(
                    optim_d,
                    total_steps=self.hparams.total_steps,
                    **self.hparams.cfg.scheduler.params_disc
                ),
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            }
            return [optim_g, optim_d], [scheduler_g, scheduler_d]
        return [optim_g, optim_d], []

    def gen_mask(self, image, apply_mask: bool = False):
        mask = torch.ones_like(image)
        if apply_mask:
            sh = mask.shape
            for idx in range(sh[0]):
                for i in range(self.hparams.cfg.mask.freq_masks):
                    x_left = np.random.randint(
                        0, max(1, sh[3] - self.hparams.cfg.mask.freq_width)
                    )
                    w = np.random.randint(0, self.hparams.cfg.mask.freq_width)
                    mask[idx, :, x_left : x_left + w, :] = 0

                for i in range(self.hparams.cfg.mask.time_masks):
                    y_left = np.random.randint(
                        0, max(1, sh[2] - self.hparams.cfg.mask.time_width)
                    )
                    w = np.random.randint(0, self.hparams.cfg.mask.time_width)
                    mask[idx, :, :, y_left : y_left + w] = 0
        return mask.type_as(image)

    def forward(self, x: Tensor, mask: Tensor):
        return self.gen_A2B(x, mask)

    def shuffle_data(self, x: Tensor):
        r = torch.randperm(x.size(0))
        return x[r]

    def training_step(self, batch):
        input_A, input_B, phase_A, phase_B = batch

        grad_clip = self.hparams.cfg.optimizer.grad_clip

        optimizer_g, optimizer_d = self.optimizers()

        scheduler_g, scheduler_d = None, None
        if self.lr_schedulers():
            scheduler_g, scheduler_d = self.lr_schedulers()

        lambda_idt = self.hparams.cfg.weight.lambda_idt
        lambda_cycle_A = self.hparams.cfg.weight.lambda_cycle_A
        lambda_cycle_B = self.hparams.cfg.weight.lambda_cycle_B

        # generator
        self.toggle_optimizer(optimizer_g)

        fake_B = self.gen_A2B(input_A, self.gen_mask(input_A, True))
        cycle_A = self.gen_B2A(fake_B, self.gen_mask(fake_B, False))

        fake_A = self.gen_B2A(input_B, self.gen_mask(input_B, True))
        cycle_B = self.gen_A2B(fake_A, self.gen_mask(fake_A, False))

        adv_1_A_loss = self.cal_adv_loss(self.disc_B(fake_A), True)
        adv_1_B_loss = self.cal_adv_loss(self.disc_A(fake_B), True)
        adv_1_loss = adv_1_A_loss + adv_1_B_loss

        adv_2_A_loss = self.cal_adv_loss(self.disc_A2(cycle_A), True)
        adv_2_B_loss = self.cal_adv_loss(self.disc_B2(cycle_B), True)
        adv_2_loss = adv_2_A_loss + adv_2_B_loss

        idt_loss_A = (
            self.idt_loss(self.gen_B2A(input_A, self.gen_mask(input_A, False)), input_A)
            * lambda_idt
        )
        idt_loss_B = (
            self.idt_loss(self.gen_A2B(input_B, self.gen_mask(input_B, False)), input_B)
            * lambda_idt
        )

        cycle_loss_A = self.cycle_loss(cycle_A, input_A) * lambda_cycle_A
        cycle_loss_B = self.cycle_loss(cycle_B, input_B) * lambda_cycle_B

        g_loss = (
            adv_1_loss
            + adv_2_loss
            + cycle_loss_A
            + cycle_loss_B
            + idt_loss_A
            + idt_loss_B
        )

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("adv1_loss", adv_1_loss)
        self.log("adv2_loss", adv_2_loss)
        self.log("adv1_fakeB_discA_loss", adv_1_B_loss)
        self.log("adv1_fakeA_discB_loss", adv_1_A_loss)
        self.log("adv2_cycleA_discA2_loss", adv_2_A_loss)
        self.log("adv2_cycleB_discB2_loss", adv_2_B_loss)
        self.log("idt_loss_A", idt_loss_A)
        self.log("idt_loss_B", idt_loss_B)
        self.log("cycle_loss_A", cycle_loss_A)
        self.log("cycle_loss_B", cycle_loss_B)

        self.manual_backward(g_loss)
        if grad_clip:
            self.clip_gradients(
                optimizer_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm"
            )
        optimizer_g.step()
        if scheduler_g:
            scheduler_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # discriminator
        self.toggle_optimizer(optimizer_d)

        # take 1/2 batch size with previous generated images.
        fake_A = self.fake_A_pool.query(fake_A)
        fake_B = self.fake_B_pool.query(fake_B)
        cycle_A = self.cycle_A_pool.query(cycle_A)
        cycle_B = self.cycle_B_pool.query(cycle_B)

        d_A_real_loss = self.cal_adv_loss(self.disc_A(input_B), True)
        d_B_fake_inp = self.disc_A(fake_B.detach())
        d_A_fake_loss = self.cal_adv_loss(d_B_fake_inp, False)
        d_A_loss = (d_A_real_loss + d_A_fake_loss) * 0.5

        d_B_real_loss = self.cal_adv_loss(self.disc_B(input_A), True)
        d_A_fake_inp = self.disc_B(fake_A.detach())
        d_B_fake_loss = self.cal_adv_loss(d_A_fake_inp, False)
        d_B_loss = (d_B_real_loss + d_B_fake_loss) * 0.5

        d_A2_real_loss = self.cal_adv_loss(self.disc_A2(input_A), True)
        d_A2_fake_inp = self.disc_A2(cycle_A.detach())
        d_A2_fake_loss = self.cal_adv_loss(d_A2_fake_inp, False)
        d_A2_loss = (d_A2_real_loss + d_A2_fake_loss) * 0.5

        d_B2_real_loss = self.cal_adv_loss(self.disc_B2(input_B), True)
        d_B2_fake_inp = self.disc_B2(cycle_B.detach())
        d_B2_fake_loss = self.cal_adv_loss(d_B2_fake_inp, False)
        d_B2_loss = (d_B2_real_loss + d_B2_fake_loss) * 0.5

        # d_loss = (
        #     d_A_loss + d_B_loss + d_A2_loss + d_B2_loss
        # )

        # self.log("d_loss", d_loss, prog_bar=True)
        self.log("d_A_loss", d_A_loss, prog_bar=True)
        self.log("d_B_loss", d_B_loss, prog_bar=True)
        self.log("d_A2_loss", d_A2_loss, prog_bar=True)
        self.log("d_B2_loss", d_B2_loss, prog_bar=True)
        self.log("d_A_accuracy", self.cal_accuracy(d_A_fake_inp))
        self.log("d_B_accuracy", self.cal_accuracy(d_B_fake_inp))
        self.log("d_A2_accuracy", self.cal_accuracy(d_A2_fake_inp))
        self.log("d_B2_accuracy", self.cal_accuracy(d_B2_fake_inp))

        # update individually to make it's focus on their task
        self.manual_backward(d_A_loss)
        self.manual_backward(d_B_loss)
        self.manual_backward(d_A2_loss)
        self.manual_backward(d_B2_loss)
        optimizer_d.step()

        if grad_clip:
            self.clip_gradients(
                optimizer_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm"
            )
        optimizer_d.step()
        if scheduler_d:
            scheduler_d.step()
        optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

    def plot_wav(self):
        with torch.inference_mode():
            mag_coms = self.mag_coms.type_as(self.gen_A2B.downsample.model[0].weight)
            fake_magnitude_B = self.gen_A2B(mag_coms, self.gen_mask(mag_coms, False))
            mags = torch.cat([i for i in fake_magnitude_B], dim=2)[
                :, :, : self.phase.size(2)
            ]

            cal_istft = T.InverseSpectrogram(**self.hparams.istft_params).cpu()
            wav = cal_istft(mags.cpu() + torch.exp(self.phase.cpu() * 1j))
            torchaudio.save("temporary.wav", wav, self.sr)
            data = [[wandb.Audio("temporary.wav", caption="Clean -> Noisy")]]
            self.logger.log_table(
                key="AudioTable", columns=["Generated_Audio"], data=data
            )

    def on_train_epoch_end(self):
        if self.cfg.hparams.log_wav:
            self.plot_wav()

    # def on_train_epoch_end(self):
    #     mag_A = torch.stack([i[0] for i in self.training_output], dim=0).type_as(
    #         self.gen_A2B.downsample.model[0].weight
    #     )
    #     mag_B = torch.stack([i[1] for i in self.training_output], dim=0).type_as(
    #         self.gen_A2B.downsample.model[0].weight
    #     )

    #     with torch.inference_mode():
    #         mask = self.gen_mask(mag_A, False)

    #         fake_B = self.gen_A2B(mag_A, mask)
    #         cycle_A = self.gen_B2A(fake_B, mask)

    #         fake_A = self.gen_B2A(mag_B, mask)
    #         cycle_B = self.gen_A2B(fake_A, mask)

    #     def normalize(x):
    #         mn = x.min()
    #         mx = x.max()
    #         return (x + 1) * 0.5 - (mx - mn) + mn

    #     mag_A, fake_A, cycle_A, mag_B, fake_B, cycle_B = list(
    #         map(
    #             lambda x: normalize(x.cpu()),
    #             [mag_A, fake_A, cycle_A, mag_B, fake_B, cycle_B],
    #         )
    #     )

    #     mag_A, fake_A, cycle_A, mag_B, fake_B, cycle_B = list(
    #         map(
    #             lambda x: torch.from_numpy(librosa.amplitude_to_db(x.numpy())),
    #             [mag_A, fake_A, cycle_A, mag_B, fake_B, cycle_B],
    #         )
    #     )

    #     A = torch.cat([mag_A, fake_A, cycle_A], dim=0)
    #     B = torch.cat([mag_B, fake_B, cycle_B], dim=0)

    #     grid_A = make_grid(A, nrow=mag_A.size(0), padding=5)
    #     grid_B = make_grid(B, nrow=mag_A.size(0), padding=5)

    #     self.logger.log_image("clean_real_fake_cycle", [grid_A])
    #     self.logger.log_image("noisy_real_fake_cycle", [grid_B])

    #     self.training_output.clear()

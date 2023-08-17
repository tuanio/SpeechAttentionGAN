from torch import Tensor
import lightning as L
import torchaudio.transforms as T
from itertools import chain
from .generator import AttentionGuideGenerator
from .discriminator import PatchGAN


class MagnitudeAttentionGAN(L.LightningModule):
    """
    call A, B as two domain considering
    in term of speech: A is clean speech and B is noisy speech
    # this class work for magnitude only
    """

    def __init__(self, cfg, stft_params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.define_generators()
        self.define_discriminators()
        self.define_loss()

        self.istft = T.InverseSpectrogram(**stft_params)

    def define_generators(self):
        self.gen_A2B = AttentionGuideGenerator(**self.hparams.cfg.generator)
        self.gen_B2A = AttentionGuideGenerator(**self.hparams.cfg.generator)

    def define_discriminators(self):
        # first adversarial loss
        self.disc_A = PatchGAN(**self.hparams.cfg.discriminator)
        self.disc_B = PatchGAN(**self.hparams.cfg.discriminator)

        # for second adversarial loss
        self.disc_A2 = PatchGAN(**self.hparams.cfg.discriminator)
        self.disc_B2 = PatchGAN(**self.hparams.cfg.discriminator)

    def define_loss(self):
        self.idt_loss = nn.L1Loss()
        self.cycle_loss = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()

    def cal_adv_loss(self, predict, is_real):
        if is_real:
            label = torch.ones_like(predict).type_as(predict)
        else:
            label = torch.zeros_like(predict).type_as(predict)
        return self.adv_loss(predict, label)

    def cal_accuracy(self, predict: Tensor, threshold: float = 0.5):
        return (predict.sigmoid() >= threshold).sum() / torch.mul(*predict.size())

    def configure_optimizers(self):
        if self.hparams.cfg.optimizer.name == "adam":
            optimizer_class = torch.optim.Adam
        elif self.hparams.cfg.optimizer.name == "adamw":
            optimizer_class = torch.optim.AdamW

        if self.hparams.cfg.scheduler.name == "one_cycle_lr":
            scheduler_class = torch.optim.lr_scheduler.OneCycleLR
        else:
            scheduler_class = None

        g_params = [
            p.parameters() for i, p in vars(self).items() if i.startswith("gen_")
        ]
        d_params = [
            p.parameters() for i, p in vars(self).items() if i.startswith("disc_")
        ]
        optim_g = optimizer_class(chain(*g_params), **self.hparams.cfg.optimizer.params)
        optim_d = optimizer_class(chain(*d_params), **self.hparams.cfg.optimizer.params)

        if scheduler_class != None:
            scheduler_g = {
                "scheduler": scheduler_class(optim_g, **self.hparams.cfg.scheduler.params),
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            }

            scheduler_d = {
                "scheduler": scheduler_class(optim_g, **self.hparams.cfg.scheduler.params),
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
                    mask[idx, :, :, x_left : x_left + w] = 0

                for i in range(self.hparams.cfg.mask.time_masks):
                    y_left = np.random.randint(
                        0, max(1, sh[2] - self.hparams.cfg.mask.time_width)
                    )
                    w = np.random.randint(0, self.hparams.cfg.mask.time_width)
                    input_spec[idx, :, y_left : y_left + w, :] = 0
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
        if self.schedulers():
            scheduler_g, scheduler_d = self.schedulers()

        lambda_idt = self.hparams.weight.lambda_idt
        lambda_cycle_B = self.hparams.weight.lambda_cycle_A
        lambda_cycle_B = self.hparams.weight.lambda_cycle_B

        # generator
        self.toggle_optimizer(optimizer_g)

        fake_B = self.gen_A2B(input_A, self.gen_mask(input_A, True))
        cycle_A = self.gen_B2A(fake_B, self.gen_mask(fake_B, False))

        fake_A = self.gen_B2A(input_B, self.gen_mask(input_B, True))
        cycle_B = self.gen_A2B(fake_A, self.gen_mask(fake_B, False))

        adv_1_loss = self.cal_adv_loss(self.disc_B(fake_B), True) + self.cal_adv_loss(
            self.disc_A(fake_A), True
        )
        adv_2_loss = self.cal_adv_loss(self.disc_B2(cycle_B), True) + self.cal_adv_loss(
            self.disc_A2(cycle_A), True
        )
        idt_loss_A = self.idt_loss(self.gen_B2A(input_A), input_A) * lambda_idt
        idt_loss_B = self.idt_loss(self.gen_A2B(input_B), input_B) * lambda_idt

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
        self.log("idt_loss_A", idt_loss_A)
        self.log("idt_loss_B", idt_loss_B)
        self.log("cycle_loss_A", cycle_loss_A)
        self.log("cycle_loss_B", cycle_loss_B)

        self.manual_backward(g_loss)
        if not grad_clip:
            self.clip_gradients(
                optimizer_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm"
            )
        optimizer_g.step()
        if scheduler_g:
            scheduler_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # log audio

        # discriminator
        self.toggle_optimizer(optimizer_d)

        # shuffle images for diverse training data - cycle gan
        # idea come from using history generated to update gan
        fake_A = self.shuffle_data(fake_A)
        fake_B = self.shuffle_data(fake_B)
        cycle_A = self.shuffle_data(cycle_A)
        cycle_B = self.shuffle_data(cycle_B)

        d_A_real_loss = self.cal_adv_loss(self.disc_A(input_A), True)
        d_A_fake_inp = self.disc_A(fake_A.detach())
        d_A_fake_loss = self.cal_adv_loss(d_A_fake_inp, False)

        d_B_real_loss = self.cal_adv_loss(self.disc_B(input_B), True)
        d_B_fake_inp = self.disc_B(fake_B.detach())
        d_B_fake_loss = self.cal_adv_loss(d_B_fake_inp, False)

        d_A2_real_loss = self.cal_adv_loss(self.disc_A2(input_A), True)
        d_A2_fake_inp = self.disc_A2(cycle_A.detach())
        d_A2_fake_loss = self.cal_adv_loss(d_A2_fake_inp, False)

        d_B2_real_loss = self.cal_adv_loss(self.disc_B2(input_B), True)
        d_B2_fake_inp = self.disc_B2(cycle_B.detach())
        d_B2_fake_loss = self.cal_adv_loss(d_B2_fake_inp, False)

        d_loss = (
            d_A_real_loss
            + d_A_fake_loss
            + d_B_real_loss
            + d_B_fake_loss
            + d_A2_real_loss
            + d_A2_fake_loss
            + d_B2_real_loss
            + d_B2_fake_loss
        )

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("d_A_real_loss", d_A_real_loss)
        self.log("d_A_fake_loss", d_A_fake_loss)
        self.log("d_B_real_loss", d_B_real_loss)
        self.log("d_B_fake_loss", d_B_fake_loss)
        self.log("d_A2_real_loss", d_A2_real_loss)
        self.log("d_A2_fake_loss", d_A2_fake_loss)
        self.log("d_B2_real_loss", d_B2_real_loss)
        self.log("d_B2_fake_loss", d_B2_fake_loss)
        self.log("d_A_accuracy", self.cal_accuracy(d_A_fake_inp))
        self.log("d_B_accuracy", self.cal_accuracy(d_B_fake_inp))
        self.log("d_A2_accuracy", self.cal_accuracy(d_A2_fake_inp))
        self.log("d_B2_accuracy", self.cal_accuracy(d_B2_fake_inp))

        self.manual_backward(d_loss)
        if not grad_clip:
            self.clip_gradients(
                optimizer_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm"
            )
        optimizer_d.step()
        if scheduler_d:
            scheduler_d.step()
        optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

    def on_train_epoch_end(self):
        pass

import torch
from data import SpeechDataModule
from model import MagnitudeAttentionGAN, AttentionGuideGenerator, PatchGAN
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="conf", config_name="default_attn_gan")
def main(cfg: DictConfig):
    print("Setup datamodule...")
    dm = SpeechDataModule(cfg.dm)

    total_steps = len(dm.train_dataloader()) * cfg.trainer.max_epochs

    print("Setup GAN model")
    model = MagnitudeAttentionGAN(
        cfg.model, total_steps=total_steps, istft_params=cfg.feature.istft_params
    )

    print("Setup logger...")
    logger = None
    if cfg.logger.wandb.have:
        logger = L.loggers.wandb.WandbLogger(**cfg.logger.wandb.params)

    print("Setup trainer...")
    trainer = L.Trainer(logger=logger, **cfg.trainer)

    if cfg.exp.train:
        print("Start training...")
        trainer.fit(model, datamodule=dm)

    if cfg.exp.test:
        print("Start testing...")
        trainer.test(model, datamodule=dm)

    if cfg.exp.predict:
        model = MagnitudeAttentionGAN.load_from_checkpoint(cfg.exp.ckpt_path)
        model.eval()
        model.plot_wav(False)


if __name__ == "__main__":
    main()

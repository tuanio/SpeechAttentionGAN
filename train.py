import torch
from data import SpeechDataModule
from model import MagnitudeAttentionGAN
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="conf", config_name="default_attn_gan")
def main(cfg: DictConfig):
    dm = SpeechDataModule(cfg.dm)
    model = MagnitudeAttentionGAN(cfg.model, istft_params=cfg.feature.istft_params)
    logger = L.logger.wandb.WandbLogger(**cfg.logger.wandb)
    trainer = L.Trainer(logger=logger, **cfg.trainer)

    if cfg.stage.train:
        print("Start training...")
        trainer.fit(model, datamodule=dm)

    if cfg.stage.test:
        print("Start testing...")
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()

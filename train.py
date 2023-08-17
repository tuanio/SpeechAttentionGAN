import torch
from data import SpeechDataModule
from model import MagnitudeAttentionGAN, AttentionGuideGenerator
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="conf", config_name="default_attn_gan")
def main(cfg: DictConfig):

    dm = SpeechDataModule(cfg.dm)
    model = MagnitudeAttentionGAN(cfg.model, istft_params=cfg.feature.istft_params)

    logger = None
    if cfg.logger.wandb.have:
        logger = L.loggers.wandb.WandbLogger(**cfg.logger.wandb.params)

    trainer = L.Trainer(logger=logger, **cfg.trainer)

    if cfg.exp.train:
        print("Start training...")
        trainer.fit(model, datamodule=dm)

    if cfg.exp.test:
        print("Start testing...")
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()

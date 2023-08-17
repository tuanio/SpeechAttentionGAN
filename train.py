import torch
from data import SpeechDataModule
from model import MagnitudeAttentionGAN, AttentionGuideGenerator
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="conf", config_name="default_attn_gan")
def main(cfg: DictConfig):

    # x = torch.rand(4, 1, 129, 128)
    # m = torch.rand(4, 1, 129, 128)

    # gen = AttentionGuideGenerator(**cfg.model.generator)

    # o = gen(x, m)

    # print(o.shape)

    torch.set_float32_matmul_precision('medium' | 'high')

    dm = SpeechDataModule(cfg.dm)

    total_steps = len(dm.train_dataloader()) * cfg.trainer.max_epochs

    model = MagnitudeAttentionGAN(
        cfg.model, total_steps=total_steps, istft_params=cfg.feature.istft_params
    )

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

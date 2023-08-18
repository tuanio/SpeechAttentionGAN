import torch
from data import SpeechDataModule
from model import MagnitudeAttentionGAN, AttentionGuideGenerator, PatchGAN
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="conf", config_name="default_attn_gan")
def main(cfg: DictConfig):
    
    x = torch.rand(4, 1, 129, 128)
    m = torch.rand(4, 1, 129, 128)

    gen = AttentionGuideGenerator(**cfg.model.generator)
    print(gen)
    print(sum(p.numel() for p in gen.parameters()))

    o = gen(x, m)

    print(o.shape)

    # dis = PatchGAN(**cfg.model.discriminator)
    # print(dis)

if __name__ == "__main__":
    main()

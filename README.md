# SpeechAttentionGAN

Pytorch, Lightning Pytorch implementation of AttentionGAN paper that compatible with Speech, use for Speech translation task.


## Note
- reflection pad to prevent checkboard artifacts and save information of edge in images
- if output go through tanh activation, not need to use instance norm before tanh
- normalize magnitude to [-1, 1] range to match with tanh activation of magnitude generation head
- phase generation neither need go through tanh activation nor normalize [-1, 1] (in case it's complex value)
- in case of phase is just angle (meaning that take float value of phase instead of complex, we still can generate through [-1, 1] then tanh activation)

## Solution to prevent "mode collapse"
* check meaning that it's solve and stride mean that not use 
- [ ] <strike> Backward individual D loss instead of add them up </strike> (split to individual, make each Ds do their old task)
- [ ] <strike> Vocoder MelGAN instead </strike> (cannot, because MelGAN does not good to model noisy - [Colab](https://colab.research.google.com/drive/191ul8y_rLHfPH-oceNnU9cxgpZ4tbUFS?usp=sharing))
- [ ] Remove shuffle generated data
- [ ] Using image pool like author's code 
    - Part 2.3 of [https://arxiv.org/abs/1612.07828](https://arxiv.org/pdf/1612.07828.pdf)
- [ ] Edit the generator
- [ ] Edit the discriminator
- [ ] Deep Conformer Generator modeling Spectrogram/MelSpectrogram (sequence modeling instead of just fix length frames)
- Adversarial Loss:
    - [ ] <strike> BCEWithLogitsLoss </strike> (very unstable, in my experiments, the loss keep increasing)
    - [ ] MSELoss (author loss, Fabian advice that it, also many paper and code use this instead) 
- Feature input:
    - [ ] Whole feature (modeling the whole feature sequence)
    - [ ] Fix-length chunks
    - [ ] Normalize to [-1, 1] anh Tanh activation
    - [ ] Raw feature with Linear activation
- [ ] Gradient Penalty
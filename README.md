# SpeechAttentionGAN

Pytorch, Lightning Pytorch implementation of AttentionGAN paper that compatible with Speech, use for Speech translation task.


## Note
- reflection pad to prevent checkboard artifacts and save information of edge in images
- if output go through tanh activation, not need to use instance norm before tanh
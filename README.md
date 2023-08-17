# SpeechAttentionGAN

Pytorch, Lightning Pytorch implementation of AttentionGAN paper that compatible with Speech, use for Speech translation task.


## Note
- reflection pad to prevent checkboard artifacts and save information of edge in images
- if output go through tanh activation, not need to use instance norm before tanh
- normalize magnitude to [-1, 1] range to match with tanh activation of magnitude generation head
- phase generation neither need go through tanh activation nor normalize [-1, 1] (in case it's complex value)
- in case of phase is just angle (meaning that take float value of phase instead of complex, we still can generate through [-1, 1] then tanh activation)
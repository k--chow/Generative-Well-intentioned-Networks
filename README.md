# Generative Well-intentioned Networks
This is an implementation of the [GWIN paper](https://papers.nips.cc/paper/9467-generative-well-intentioned-networks), which maps the uncertain distribution of a probabilistic classifier to its more confident distribution. The use case here is demonstrated with a bayesian neural network classifier on MNIST with a WGAN-GP. We map samples of an certainty rejection threshold from our classifier to new samples generated by a WGAN-GP that has a critic trained to discern uncertain and confident samples. 

Some cherry-picked examples (left is original, right is new sample):

![](https://paper-attachments.dropbox.com/s_A6985D79CD6F21CF8E50B11099C04B7FAA055DA2D1D99DE6617EDAE393B1E447_1597882191675_Screen+Shot+2020-08-19+at+8.06.46+PM.png)

![](https://paper-attachments.dropbox.com/s_A6985D79CD6F21CF8E50B11099C04B7FAA055DA2D1D99DE6617EDAE393B1E447_1597882197698_Screen+Shot+2020-08-19+at+8.06.58+PM.png)


 
Note that there are some differences in the paper and my code which my account for performance differences, especially regarding GAN training. I have tried to mimic the paper closely but there may still be small differences that have large impact 

- We normalize data -1 to 1 to use a tanh rather than a sigmoid in the generator
- We use batch norm in the generator (from my limited experience trains faster) and layer normalization in the critic ([noted in the original WGAN-GP paper](https://arxiv.org/abs/1704.00028))

Code is referenced and inspired from (WGAN-GP) https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py and (tensorflow probability BNN) https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py




# References

- [1] GWIN paper https://papers.nips.cc/paper/9467-generative-well-intentioned-networks
- [2] WGAN-GP paper https://arxiv.org/abs/1704.00028


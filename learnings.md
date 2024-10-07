# Lessons

Training SAEs is not like training typical models.
There's a very strong sparsity constraint, and reconstruction loss might not affect the actual model's loss too badly.
As a result, you need different tools for logging and measuring training progress.

One tool is Pareto-like curve is measuring the change in model loss (LM or ViT) with respect to the L0 norm of the SAE.
As L0 norm increases (less sparse), the delta model loss should also decrease (higher delta indicates higher increase in loss as a result of using the SAE representation).

## Sanity Check Checklist

1. Loss goes down during training.
2. Validation loss is not significantly worse than traing loss.
3. When you plot the pareto frontier of L2 reconstruction loss and L0 (inverse of sparsity), then models with higher L0 achieve lower L2 reconstruction loss.
4. "Better" models (architectures, hyperparameters, data, etc) are on the bottom-left of the reconstruction loss-sparsity tradeoff graph. For instance, for a given architecture, *wider* models trained for the same number of examples should be better--they have more expressive power and have been trained with more FLOPs (because the model is bigger).


## Numbers

Sometimes we just need to see lots of numbers to develop intuition.

Discover-Then-Name trains for 200 epochs, resampling dead weights every 10 epochs.
They do hyperparameter sweeps over a validation set (of activations):

* learning rate: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
* lambda: [3e-5, 1.5e-4, 3e-4, 1.5e-3, 3e-3]
* expansion factor: [2, 4, 8]

They chose the best hyperparamters based on ImageNet zeroshot performance using reconstructions.

## Tricks

* Normalize the activations to have unit norms. (done)
* Use a linear warmup for the sparsity coefficient. Gemma Scope linearly increases from 0 over 10K training steps. (done)
* Initialize W_dec with He-uniform and rescale each latent vector to have unit norm. Then initialize W_enc as the transpose of W_dec. Initialize biases to 0 vectors. (done)
* You only need to train for [1 epoch](https://github.com/ai-safety-foundation/sparse_autoencoder/blob/b6ba6cb7c90372cb5462855c21e5f52fc9130557/sparse_autoencoder/train/pipeline.py#L216). On the other hand, the Discover-then-Name paper trains for [200 epochs](https://github.com/neuroexplicit-saar/discover-then-name?tab=readme-ov-file#train-the-sae) (look for `--num_epochs 200`).
* Gemma Scope paper suggests that bfloat16 is good enough for SAE weights (i.e., can use half-precision for SAE training; Section 4.7). Note that activations were still stored in fp32.



# Sparse Autoencoders

This repo trains sparse autoencoders (SAEs) on vision transformers (ViTs); specifically, we use SAEs to discover traits in [BioCLIP](https://imageomics.github.io/bioclip/).
To train an SAE on an existing, pre-trained ViT, there are three steps, each of which are supported in this repo:

1. Get ViT activations on a new dataset (`get_activations.py`).
2. Train an SAE on the activations (`train_sae.py`).
3. Look at the discovered features to understand what's going on (`notebooks/explore.py`, using [marimo](https://marimo.io/)).


## Reference Implementation

[This GitHub repo](https://github.com/HugoFry/mats_sae_training_for_ViTs) works for training SAEs on ViTs.
Since my code doesn't work, I am re-implementing it step-by-step in Jax, checking for numerical equivalence (with errors due to floating-point math).

To run some tests comparing against the reference implementation, run:

```sh
CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cpu uv run pytest tests/test_reference.py^C
```

Specifically, provide a GPU for the PyTorch-only tests, and specify that Jax has to run on the CPU to make sure numerical equivalence is more likely.

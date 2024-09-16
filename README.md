# Sparse Autoencoders

This repo trains sparse autoencoders (SAEs) on vision transformers (ViTs); specifically, we use SAEs to discover traits in [BioCLIP](https://imageomics.github.io/bioclip/).
To train an SAE on an existing, pre-trained ViT, there are three steps, each of which are supported in this repo:

1. Get ViT activations on a new dataset (`get_activations.py`).
2. Train an SAE on the activations (`train_sae.py`).
3. Look at the discovered features to understand what's going on (`notebooks/explore.py`, using [marimo](https://marimo.io/)).



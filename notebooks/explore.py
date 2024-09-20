import marimo

__generated_with = "0.8.17"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import datasets
    import sax.nn
    import sax.helpers
    import jax
    import time
    import jax.numpy as jnp
    import beartype
    from jaxtyping import Array, Float, Int, jaxtyped
    return (
        Array,
        Float,
        Int,
        beartype,
        datasets,
        jax,
        jaxtyped,
        jnp,
        mo,
        np,
        sax,
        time,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Explore SAEs

        This notebook helps you explore different trained sparse autoencoders (SAEs) and understand what features are found.
        It's similar to this [sae-explorer](https://sae-explorer.streamlit.app/).
        """
    )
    return


@app.cell
def __(Array, Int, datasets, indices):
    class Images:
        def __init__(self, indices: Int[Array, "n_features 16"]):
            self._images = datasets.load_dataset(
                "ILSVRC/imagenet-1k", split="train", trust_remote_code=True
            )
            self._indices = indices

        def get_best(self, n: int):
            idx = self._indices[n].tolist()
            images = [self._images[i]["image"] for i in idx]
            # resize
            images = [image.copy() for image in images]
            for image in images:
                image.thumbnail((128, 128))
            return images


    images = Images(indices)
    return Images, images


@app.cell
def __(mo):
    neuron_picker = mo.ui.text(value="0")
    return neuron_picker,


@app.cell
def __(mo):
    def grid(images, *, n_cols: int = 8):
        rows = [images[i : i + n_cols] for i in range(0, len(images), n_cols)]
        return mo.vstack([mo.hstack(row) for row in rows], gap=0)
    return grid,


@app.cell
def __(grid, images, mo, neuron_picker):
    mo.md(f"""
    Neuron: {neuron_picker} (Choose a number between 0 and 49152)

    # Top 16 Activating Images

    {grid(images.get_best(int(neuron_picker.value)))}
    {neuron_picker.value}
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        How do we find the top N images for each SAE feature?
        At the end of the algorithm, I want to have an $D \times N$ matrix, where $D$ is the number of SAE features. 
        Position $i, j$ in the matrix should contain index of the image that has the $i$-th highest activation of feature $j$.
        Furthermore, the algorithm should be $O(I)$ where $I$ is the number of images.

        1. For each image activation, feed it into the SAE and get a feature activation vector $F \in \mathbb{R}^D$.
        2. For each feature $j$, insert the image index into the matrix if $F_j$ is larger.

        This implies that I need a $D \times N \times 2$ matrix.
        """
    )
    return


@app.cell
def __(Array, Float, Int, beartype, jax, jaxtyped, jnp, np, sax, time):
    @jaxtyped(typechecker=beartype.beartype)
    @jax.jit
    def push(
        scores: Float[Array, "n_features 16"],
        indices: Int[Array, "n_features 16"],
        incoming_scores: Float[Array, "n_features"],
        incoming_index: int,
    ) -> tuple[Float[Array, "n_features 16"], Int[Array, "n_features 16"]]:
        at = jax.vmap(jnp.searchsorted)(scores, incoming_scores)
        scores = jax.vmap(jnp.insert)(scores, at, incoming_scores)[:, :-1]
        indices = jax.vmap(jnp.insert, in_axes=(0, 0, None))(indices, at, incoming_index)[:, :-1]
        return scores, indices


    def sort_image_feature_sims(activations_path, ckpt_path, *, batch_size: int = 4096):
        sae = sax.helpers.load(ckpt_path, sax.nn.ReparamInvariantReluSAE)

        n_activations = 1281167

        shape = (n_activations, 12, 768)
        data = np.memmap(activations_path, dtype=np.float32, mode="r", shape=shape)

        scores = jnp.full((49152, 16), jnp.inf)
        indices = jnp.full((49152, 16), -1)

        start_time = time.time()
        start = 0
        while True:
            end = min(start + batch_size, n_activations)
            
            batch = jnp.array(data[start:end, -1])
            _, f_x = jax.vmap(sae)(batch)
            for i in range(end - start):
                scores, indices = push(scores, indices, -f_x[i], start + i)
            # Progress meter.
            print(f"{end}/{n_activations} ({end/n_activations*100:.1f}%) at {end / (time.time() - start_time):.1f} ex/s")
            start = end
            
            if start >= n_activations:
                break

        return scores, indices


    _, indices = sort_image_feature_sims(
        "/local/scratch/stevens.994/datasets/sax/ViT-B-16_openai/activations-imagenet.bin",
        "/home/stevens.994/projects/sax/checkpoints/4504f6a348af48fa84f2f118.eqx",
    )
    return indices, push, sort_image_feature_sims


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

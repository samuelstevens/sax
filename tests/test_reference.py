import sys

sys.path.insert(0, "/home/stevens.994/projects/sax/vendored/mats_sae_training_for_ViTs")


import beartype
import datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float, jaxtyped
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader
from vit_sae_analysis import dashboard_fns

import sax
import sax.nn

atol = 1e-5
rtol = 1e-5


def test_smoke():
    pass


def get_ref_inputs(ref_model, images):
    return ref_model.processor(
        images=images, text="", return_tensors="pt", padding=True
    )


def get_my_inputs(my_model, images):
    my_img_transform = my_model.make_img_transform()

    def transform(image):
        image = image.convert("RGB")
        image = my_img_transform(image)
        return image

    return torch.stack([transform(image) for image in images])


def where_diff(a, b, atol=atol, rtol=1e-7):
    return jnp.abs(a - b) > atol + rtol * jnp.abs(b)


@torch.inference_mode()
def test_model_preprocessing():
    """
    Tests that the majority of the images are preprocessed the same by both implementations.
    """
    batch_size = 128
    dataset = datasets.load_dataset("ILSVRC/imagenet-1k", split="train")
    images = dataset[:batch_size]["image"]

    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"
    cfg = torch.load(ckpt_path, map_location="cpu")["cfg"]

    loader = ViTSparseAutoencoderSessionloader(cfg)
    ref_model = loader.get_model(cfg.model_name)
    ref_inputs = get_ref_inputs(ref_model, images)

    my_model = sax.load_vision_backbone("open-clip", "ViT-L-14/openai").to(cfg.device)
    my_inputs = get_my_inputs(my_model, images)

    equal = (
        (my_inputs == ref_inputs["pixel_values"])
        .type(torch.float64)
        .mean(axis=(1, 2, 3))
    )
    assert (equal >= 1.0).sum() >= 116


@torch.inference_mode()
def test_get_model_activations_ref_inputs():
    batch_size = 32
    dataset = datasets.load_dataset("ILSVRC/imagenet-1k", split="train")
    images = dataset[:batch_size]["image"]

    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"
    cfg = torch.load(ckpt_path, map_location="cpu")["cfg"]

    loader = ViTSparseAutoencoderSessionloader(cfg)
    ref_model = loader.get_model(cfg.model_name)
    ref_model.to(cfg.device)
    ref_inputs = get_ref_inputs(ref_model, images).to(cfg.device)
    ref_activations = dashboard_fns.get_model_activations(ref_model, ref_inputs, cfg)
    # [batch, 1024]
    ref_activations = ref_activations.cpu().numpy()

    my_model = sax.load_vision_backbone("open-clip", "ViT-L-14/openai").to(cfg.device)
    recorder = sax.Recorder(my_model)
    my_model.img_encode(ref_inputs["pixel_values"])
    # [batch, 1024]
    my_activations = recorder.activations.numpy()[:, -2, 0, :]

    np.testing.assert_allclose(ref_activations, my_activations, atol=atol)


@torch.inference_mode()
def test_get_model_activations_my_inputs():
    batch_size = 32
    dataset = datasets.load_dataset("ILSVRC/imagenet-1k", split="train")
    images = dataset[:batch_size]["image"]

    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"
    cfg = torch.load(ckpt_path, map_location="cpu")["cfg"]

    my_model = sax.load_vision_backbone("open-clip", "ViT-L-14/openai").to(cfg.device)
    recorder = sax.Recorder(my_model)
    my_inputs = get_my_inputs(my_model, images).to(cfg.device)
    my_model.img_encode(my_inputs)
    # [batch, 1024]
    my_activations = recorder.activations.numpy()[:, -2, 0, :]

    loader = ViTSparseAutoencoderSessionloader(cfg)
    ref_model = loader.get_model(cfg.model_name)
    ref_model.to(cfg.device)
    ref_inputs = get_ref_inputs(ref_model, images).to(cfg.device)
    ref_inputs["pixel_values"] = my_inputs
    ref_activations = dashboard_fns.get_model_activations(ref_model, ref_inputs, cfg)
    # [batch, 1024]
    ref_activations = ref_activations.cpu().numpy()

    np.testing.assert_allclose(ref_activations, my_activations, atol=atol)


@torch.inference_mode()
def test_sae_forward_pass_eval():
    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"

    loaded_object = torch.load(ckpt_path, map_location="cpu")
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    cfg.device = torch.device("cpu")

    ref_sae = SparseAutoencoder(cfg)
    ref_sae.load_state_dict(state_dict)
    ref_sae.to("cpu")
    ref_sae.eval()

    my_sae = sax.nn.HugoFrySAE(
        d_in=1024, d_hidden=1024 * 64, key=jax.random.key(seed=0)
    ).from_torch(ref_sae)

    # Check that from_torch worked.
    np.testing.assert_allclose(my_sae.w_enc.T, ref_sae.W_enc.cpu().numpy())
    np.testing.assert_allclose(my_sae.w_dec.T, ref_sae.W_dec.cpu().numpy())

    rng = np.random.default_rng(seed=12)
    for _ in range(10):
        inputs_np = rng.normal(size=(32, 1024)).astype(np.float64)
        my_out, my_acts = jax.vmap(my_sae)(inputs_np)
        ref_out, ref_acts, *_ = ref_sae(torch.from_numpy(inputs_np))

        np.testing.assert_allclose(
            my_acts, ref_acts.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(my_out, ref_out.cpu().numpy(), atol=atol, rtol=rtol)


def test_matmuls():
    rng = np.random.default_rng(seed=32)

    batch_size = 32
    d_in = 256
    d_out = 2**16

    for _ in range(10):
        inputs_np = rng.normal(size=(batch_size, d_in)).astype(np.float64)

        torch_mat = torch.rand((d_out, d_in), dtype=torch.float64) - 0.5
        jax_mat = jnp.array(torch_mat.numpy())
        np.testing.assert_allclose(jax_mat, torch_mat.numpy())

        torch_out = torch.from_numpy(inputs_np) @ torch_mat.T
        jax_out = inputs_np @ jax_mat.T
        np.testing.assert_allclose(jax_out, torch_out.numpy(), atol=atol, rtol=rtol)


@pytest.mark.xfail(reason="We don't pass dead_neuron_mask to the reference model yet.")
def test_forward_pass_train():
    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"

    loaded_object = torch.load(ckpt_path, map_location="cpu")
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    cfg.device = torch.device("cpu")

    ref_sae = SparseAutoencoder(cfg)
    ref_sae.load_state_dict(state_dict)
    ref_sae.to("cpu")
    ref_sae.train()

    my_sae = sax.nn.HugoFrySAE(
        d_in=1024, d_hidden=1024 * 64, key=jax.random.key(seed=0)
    ).from_torch(ref_sae)

    # Check that from_torch worked.
    np.testing.assert_allclose(my_sae.w_enc.T, ref_sae.W_enc.detach().cpu().numpy())
    np.testing.assert_allclose(my_sae.w_dec.T, ref_sae.W_dec.detach().cpu().numpy())

    rng = np.random.default_rng(seed=12)
    for _ in range(10):
        inputs_np = rng.normal(size=(32, 1024)).astype(np.float64)
        my_out, my_acts = jax.vmap(my_sae)(inputs_np)
        ref_out, ref_acts, *_ = ref_sae(torch.from_numpy(inputs_np))

        np.testing.assert_allclose(
            my_acts, ref_acts.detach().cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_out, ref_out.detach().cpu().numpy(), atol=atol, rtol=rtol
        )


def test_calc_loss_train():
    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"

    loaded_object = torch.load(ckpt_path, map_location="cpu")
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    cfg.device = torch.device("cpu")

    ref_sae = SparseAutoencoder(cfg)
    ref_sae.load_state_dict(state_dict)
    ref_sae.to("cpu")
    ref_sae.train()

    my_sae = sax.nn.HugoFrySAE(
        d_in=1024, d_hidden=1024 * 64, key=jax.random.key(seed=0)
    ).from_torch(ref_sae)

    rng = np.random.default_rng(seed=12)
    for _ in range(10):
        inputs_np = rng.normal(size=(32, 1024)).astype(np.float64)
        my_loss = my_sae.loss(
            my_sae, jnp.array(inputs_np), sparsity_coeff=jnp.array(cfg.l1_coefficient)
        )
        _, _, ref_loss, ref_mse_loss, ref_l1_loss, _ = ref_sae(
            torch.from_numpy(inputs_np), torch.zeros(1024 * 64, dtype=bool)
        )

        # MSE loss
        np.testing.assert_allclose(
            my_loss.reconstruction,
            ref_mse_loss.detach().cpu().numpy(),
            atol=atol,
            rtol=rtol,
        )
        # L1 loss
        np.testing.assert_allclose(
            my_loss.sparsity,
            ref_l1_loss.detach().cpu().numpy(),
            atol=atol,
            rtol=rtol,
        )
        # Sum of loss terms
        np.testing.assert_allclose(
            my_loss.loss,
            ref_loss.detach().cpu().numpy(),
            atol=atol,
            rtol=rtol,
        )


def test_backward_pass_with_parallel_grads():
    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"

    loaded_object = torch.load(ckpt_path, map_location="cpu")
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    cfg.device = torch.device("cpu")

    ref_sae = SparseAutoencoder(cfg)
    ref_sae.load_state_dict(state_dict)
    ref_sae.to("cpu")
    ref_sae.train()

    my_sae = sax.nn.HugoFrySAE(
        d_in=1024, d_hidden=1024 * 64, key=jax.random.key(seed=0)
    ).from_torch(ref_sae)

    @jaxtyped(typechecker=beartype.beartype)
    def _compute_loss(
        model: eqx.Module, batch: Float[Array, "batch d_model"]
    ) -> tuple[Float[Array, ""], sax.nn.Loss]:
        """Compute loss. Return the complete loss (loss.loss) as the first term for optimization and the pytree `loss` for metrics."""
        loss = model.loss(model, batch, jnp.array(cfg.l1_coefficient))
        return loss.loss, loss

    # Only used to clear the .grad attributes.
    dummy_opt = torch.optim.SGD(ref_sae.parameters())

    rng = np.random.default_rng(seed=12)
    for _ in range(10):
        inputs_np = rng.normal(size=(32, 1024)).astype(np.float64)
        (_, my_loss), my_grads = eqx.filter_value_and_grad(_compute_loss, has_aux=True)(
            my_sae, jnp.array(inputs_np)
        )

        dead_mask = torch.zeros(1024 * 64, dtype=bool)
        _, _, ref_loss, _, _, _ = ref_sae(torch.from_numpy(inputs_np), dead_mask)

        # Check loss term.
        np.testing.assert_allclose(
            my_loss.loss,
            ref_loss.detach().cpu().numpy(),
            atol=atol,
            rtol=rtol,
        )

        ref_loss.backward()

        # Check gradients
        np.testing.assert_allclose(
            my_grads.b_dec, ref_sae.b_dec.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.b_enc, ref_sae.b_enc.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.w_enc.T, ref_sae.W_enc.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.w_dec.T, ref_sae.W_dec.grad.cpu().numpy(), atol=atol, rtol=rtol
        )

        # Include this so that grads are reset every loop iteration.
        dummy_opt.zero_grad()


def test_backward_pass_remove_parallel_grads():
    ckpt_path = "tests/clip-vit-large-patch14_-2_resid_65536.pt"

    loaded_object = torch.load(ckpt_path, map_location="cpu")
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    cfg.device = torch.device("cpu")

    ref_sae = SparseAutoencoder(cfg)
    ref_sae.load_state_dict(state_dict)
    ref_sae.to("cpu")
    ref_sae.train()

    my_sae = sax.nn.HugoFrySAE(
        d_in=1024, d_hidden=1024 * 64, key=jax.random.key(seed=0)
    ).from_torch(ref_sae)

    @jaxtyped(typechecker=beartype.beartype)
    def _compute_loss(
        model: eqx.Module, batch: Float[Array, "batch d_model"]
    ) -> tuple[Float[Array, ""], sax.nn.Loss]:
        """Compute loss. Return the complete loss (loss.loss) as the first term for optimization and the pytree `loss` for metrics."""
        loss = model.loss(model, batch, jnp.array(cfg.l1_coefficient))
        return loss.loss, loss

    # Only used to clear the .grad attributes.
    dummy_opt = torch.optim.SGD(ref_sae.parameters())

    rng = np.random.default_rng(seed=12)
    for _ in range(10):
        inputs_np = rng.normal(size=(32, 1024)).astype(np.float64)
        (_, my_loss), my_grads = eqx.filter_value_and_grad(_compute_loss, has_aux=True)(
            my_sae, jnp.array(inputs_np)
        )

        dead_mask = torch.zeros(1024 * 64, dtype=bool)
        _, _, ref_loss, _, _, _ = ref_sae(torch.from_numpy(inputs_np), dead_mask)

        # Check loss terms.
        np.testing.assert_allclose(
            my_loss.loss,
            ref_loss.detach().cpu().numpy(),
            atol=atol,
            rtol=rtol,
        )

        ref_loss.backward()

        # Remove parallel component of gradients.
        ref_sae.remove_gradient_parallel_to_decoder_directions()
        my_grads = my_sae.remove_parallel_grads(my_grads)

        # Check gradients
        np.testing.assert_allclose(
            my_grads.b_dec, ref_sae.b_dec.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.b_enc, ref_sae.b_enc.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.w_enc.T, ref_sae.W_enc.grad.cpu().numpy(), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            my_grads.w_dec.T, ref_sae.W_dec.grad.cpu().numpy(), atol=atol, rtol=rtol
        )

        # Include this so that grads are reset every loop iteration.
        dummy_opt.zero_grad()

"""
Gets activations for a ViT's image representation and stores them as floating-point arrays on disk so they can be used as training data for SAEs.

To get activations, we need a dataloader, a model, and a place on disk to write data to.

By default, I am interested in training SAEs on BioCLIP looking at the validation split of TreeOfLife-10M.
Because of this, I use a lot of the same infrastructure that we used for training (webdataset, open_clip_torch, etc).
"""

import dataclasses
import logging
import os

import beartype
import numpy as np
import torch
import tyro
import webdataset as wds
from jaxtyping import Float, jaxtyped

import sax

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    data_url: str = "/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/val/shard-{000000..000063}.tar"
    """where the validation split is located."""
    n_examples: int = 503199
    """number of examples in data_url."""
    imagenet: bool = False
    """whether to use imagenet for debugging rather than --data-url."""
    batch_size: int = 256
    """inference batch size."""
    log_every: int = 10
    """how often to log."""
    n_workers: int = 4
    """number of dataloader workers."""
    model_ckpt: str = "hf-hub:imageomics/BioCLIP"
    """specific open_clip model checkpoint to load."""
    d_model: int = 768
    """number of dimensions of outputs."""
    write_to: str = "/fs/scratch/PAS2136/samuelstevens/datasets/sae"
    """where to write activations."""
    suffix: str = ""
    """additional suffix for the activations file."""
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    # Slurm
    slurm: bool = False
    """whether to use slurm to run the job (with submitit)."""
    slurm_acct: str = "PAS2136"
    """slurm account."""


def filter_no_caption_or_no_image(sample):
    has_caption = any("txt" in key for key in sample)
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


@beartype.beartype
def get_dataloader(args: Args, img_transform):
    if args.imagenet:
        import datasets

        def hf_transform(example):
            example["image"] = example["image"].convert("RGB")
            example["image"] = img_transform(example["image"])
            return example

        def _collate_fn(batch):
            batch = torch.utils.data.default_collate(batch)
            return (batch["image"],)

        dataset = (
            datasets.load_dataset(
                "ILSVRC/imagenet-1k", split="train", trust_remote_code=True
            )
            .to_iterable_dataset(num_shards=args.n_workers)
            .map(hf_transform)
            .with_format("torch")
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.n_workers,
            pin_memory=True,
            persistent_workers=args.n_workers > 0,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    dataset = wds.DataPipeline(
        # at this point we have an iterator over all the shards
        wds.SimpleShardList(args.data_url),
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp"),
        wds.map_dict(image=img_transform),
        wds.to_tuple("image"),
        wds.batched(args.batch_size, partial=True),
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.n_workers,
        persistent_workers=False,
    )

    return dataloader


def fs_safe(string: str) -> str:
    return string.replace(":", "_").replace("/", "_")


@jaxtyped(typechecker=beartype.beartype)
class ShardedDiskStorage:
    dirpath: str
    pattern: str
    shard_counter: int
    shard: object

    def __init__(self, dirpath: str, digits: int, examples_per_shard: int):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

        self.pattern = f"shard-%0{digits}d.bin"
        self.shard_counter = -1

        self.examples_per_shard = examples_per_shard
        self.shard = None

        self.next()

    def write(self, data: Float[np.ndarray, "batch n_layers d_model"]):
        batch_size, n_layers, d_model = data.shape
        assert self.n_per_shard % batch_size == 0

        shape = (self.n_per_shard, n_layers, 1, d_model)

        if self.shard is None:
            self._new_shard(shape)
        elif self.pos + batch_size > self.n_per_shard:
            # Time for a new shard.
            self.finish()
            self._new_shard(shape)

        self.shard[self.pos : self.pos + batch_size] = data
        self.pos += batch_size

    def _new_shard(self, shape: tuple[int, int, int, int]):
        self.finish()

        self.shard_counter += 1
        filepath = os.path.join(self.dirpath, self.pattern % self.shard_counter)
        self.shard = np.memmap(filepath, dtype=np.float32, mode="w+", shape=shape)
        self.pos = 0

    def finish(self):
        if self.shard is not None:
            self.shard.flush()
            self.shard.close()
            del self.shard


@torch.no_grad
def get_activations(args: Args):
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA GPU found. Using CPU instead.")
        args = dataclasses.replace(args, device="cpu")

    model = sax.load_vision_backbone("open-clip", args.model_ckpt).to(args.device)
    recorder = sax.Recorder(model)
    dataloader = get_dataloader(args, model.make_img_transform())

    dirpath = os.path.join(args.write_to, fs_safe(args.model_ckpt))
    os.makedirs(dirpath, exist_ok=True)
    filename = f"activations-{args.suffix}.bin" if args.suffix else "activations.bin"
    filepath = os.path.join(dirpath, filename)
    arr = np.memmap(
        filepath,
        dtype=np.float32,
        mode="w+",
        shape=(args.n_examples, recorder.n_layers, 1, args.d_model),
    )

    start = 0
    for b, (images,) in enumerate(dataloader):
        images = images.to(args.device)
        model.img_encode(images)
        activations = recorder.activations.numpy()

        end = start + len(images)
        arr[start:end] = activations
        start = end

        recorder.reset()

        if b % args.log_every == 0:
            logger.info(
                "batch: %d, example: %d/%d (%.1f%%)",
                b,
                end,
                args.n_examples,
                end / args.n_examples * 100,
            )
            arr.flush()

    logger.info("Finished embedding %d images.", end)
    arr.flush()


@torch.no_grad
def main(args: Args):
    if args.slurm:
        import submitit

        executor = submitit.AutoExecutor(folder="logs")
        executor.update_parameters(
            timeout_min=60,
            gpus_per_node=1,
            cpus_per_task=12,
            stderr_to_stdout=True,
            slurm_account=args.slurm_acct,
        )
        job = executor.submit(get_activations, args)
        job.result()
    else:
        get_activations(args)


if __name__ == "__main__":
    main(tyro.cli(Args))

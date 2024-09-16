"""
Gets activations for a ViT's image representation and stores them as floating-point arrays on disk so they can be used as training data for SAEs.

To get activations, we need a dataloader, a model, and a place on disk to write data to.

By default, I am interested in training SAEs on BioCLIP looking at the validation split of TreeOfLife-10M.
Because of this, I use a lot of the same infrastructure that we used for training (webdataset, open_clip_torch, etc).
"""

import beartype
import dataclasses
import torch
import logging

import tyro
import webdataset as wds

import sax

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    data_url: str = "/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/val/shard-{000000..000063}.tar"
    """where the validation split is located."""
    batch_size: int = 256
    """inference batch size."""
    n_workers: int = 4
    """number of dataloader workers."""
    model_ckpt: str = "hf-hub:imageomics/BioCLIP"
    """specific open_clip model checkpoint to load."""
    write_to: str = "/fs/scratch/PAS2136/samuelstevens/datasets/sae"
    """where to write activations."""


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
    # at this point we have an iterator over all the shards
    dataset = wds.DataPipeline(
        wds.SimpleShardList(args.data_url),
        wds.shuffle(),
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.shuffle(100),
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


@torch.no_grad
def main(args: Args):
    model = sax.load_vision_backbone("open-clip", args.model_ckpt)

    dataloader = get_dataloader(args, model.make_img_transform())

    for b, (images,) in enumerate(dataloader):
        logits = model(images)
        breakpoint()


if __name__ == "__main__":
    main(tyro.cli(Args))

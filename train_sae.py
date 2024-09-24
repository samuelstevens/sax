"""
Trains a sparse autoencoder (SAE) on activations.

"""

import logging

import tyro

import sax.train

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


if __name__ == "__main__":
    sax.train.train(tyro.cli(sax.train.Args))

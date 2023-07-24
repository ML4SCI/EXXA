#!/bin/bash

### Runs a wandb sweep on the autoencoder ###

export WANDB_API_KEY=199115cad71655dbb5640225359e90bc0a91bcca

## 101 seq length
wandb agent chlab/transformer_anomaly/7srnxr38

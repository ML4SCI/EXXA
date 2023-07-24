#!/bin/bash

### Runs a wandb sweep on the autoencoder ###

export WANDB_API_KEY=199115cad71655dbb5640225359e90bc0a91bcca

## new sweep
wandb agent chlab/autoencoder_anomaly/kndf00o0

## using existing runs (BAD!)
# wandb agent chlab/autoencoder_anomaly/ux45pc7g

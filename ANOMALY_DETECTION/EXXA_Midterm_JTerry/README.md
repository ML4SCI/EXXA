# Protoplanetary Disk Anomaly Detection

<h1 align="center">
  <a href="https://app.circleci.com/pipelines/github/j-p-terry/non_keplerian_anomaly_detection"><img alt="Build" src="https://shields.api-test.nl/circleci/build/github/j-p-terry/non_keplerian_anomaly_detection?style=for-the-badge&token=4bae0fb820e3e7d4ec2352639e35d499c673d78c"></a>
  <!-- <a href="https://circleci.com/github/j-p-terry/ariel_2023"><img alt="Build" src="https://circleci.com/github/j-p-terry/ariel_2023.svg?style=svg&circle-token=be8ad696030c87630cb1f0c972fbdffdfd998a1a"></a> -->
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.0+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
  <a href="https://wandb.ai/chlab/new_transformer_anomaly?workspace=user-jpterry"><img alt="Transformer: wandb" src="https://img.shields.io/badge/Fancy Transformer-wandb-f5c142.svg?style=for-the-badge&labelColor=gray"></a>
  <a href="https://wandb.ai/chlab/transformer_anomaly?workspace=user-jpterry"><img alt="Transformer: wandb" src="https://img.shields.io/badge/Transformer-wandb-f5c142.svg?style=for-the-badge&labelColor=gray"></a>
  <a href="https://wandb.ai/chlab/autoencoder_anomaly?workspace=user-jpterry"><img alt="Autoencoder: wandb" src="https://img.shields.io/badge/Autoencoder-wandb-f5c142.svg?style=for-the-badge&labelColor=gray"></a>
  <!-- <a href="https://wandb.ai/chlab/gan_anomaly?workspace=user-jpterry"><img alt="GAN: wandb" src="https://img.shields.io/badge/GAN-wandb-f5c142.svg?style=for-the-badge&labelColor=gray"></a> -->
</h1>

This repo uses unsupervised models (transformer and autoencoder) that have been trained on line emission spectra from Keplerian protoplanetary disks. These models can detect anomalies, i.e. non-Keplerian features, in comporable observations. This allows the creation of an anomaly map.

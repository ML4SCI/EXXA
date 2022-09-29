# Finding Exoplanets with Astronomical Observations

## Jason Terry

### Google Summer of Code 2022
### ML4Sci

<h1 align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
</h1>

For a more thorough explanation, read [this blog](https://medium.com/@jason.terry47/finding-exoplanets-with-deep-learning-1d271c73e588).

Protoplanetary disks are the site of planet formation. As our observational capabilities have increased, e.g. the commisionning of ALMA, we have collected more and better data of these systems. A variety of methods have been tried to locate planets within them. This task is made difficult not only by the limited resolution and inherent difficulties of observing, but also by the fact that the planets can be deeply embedded within the somewhat opaque disk. However, regardless of their location, planets affect the gas and dust within the disk. Using line emission measurements in the infrared, we can see these disturbances in the motion. This allows us to do so-called kinematic analysis of the disk. By looking at the motion of the gas and dust within the disk, we can infer properties of the disk and any bodies within it. This method has been successful in locating at least two planets by Pinte et al. ([2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860L..13P/abstract), [2019](https://ui.adsabs.harvard.edu/abs/2019NatAs...3.1109P/abstract)). Encouraged by this success, we attempt to use kinematic data of protoplanetary disks to determine whether a given observation has a planet and, if so, locate the planet or planets.

The data used was generated in a two-part process: smoothed particle hydrodynamics simulations to model disk evolution using [PHANTOM](https://phantomsph.readthedocs.io/en/latest/index.html) followed by radiative transfer calculations that were done with [MCFOST](https://mcfost.readthedocs.io/en/latest/). 1,000 different systems were simulated under a variety of physical conditions, and models were constucted to classify a system as having no planet or having at least one planet.

- RegNet
  - [Paper](https://arxiv.org/abs/2101.00590)
  - [Original Model Code](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)
- EfficientNetV2
  - [Paper](https://arxiv.org/abs/2104.00298)
  - [Original Model Code](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)

Hyperparameter sweeps using [wandb](https://wandb.ai/site) were done.

Some training and simulating scripts can be found in another [repository](https://github.com/j-p-terry/Finding-Exoplanets-with-Astronomical-Observations). Given the size of the models and data, not all are presented, but a user can download the models and a subset of the data.

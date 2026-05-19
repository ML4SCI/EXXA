# EXXA

## Background

EXXA is a [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) project that focuses on using machine learning to characterizing forming and existing exoplanets using both synthetic and observational multi-modal data.

### Exoplanets
Exoplanets are planets that are outside of our Solar System. To date, roughly 6,000 exoplanets have been confirmed using a variety of detection methods. Identifying and characterizing exoplanets informs out theories of planet formation and may allow us to test astrobiological hypotheses.


<p align="center">
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/2/2e/ExoplanetPopulations-20170616.png"
    width="700"
    alt="Chart showing populations of known exoplanets by size and orbital period"
  /><br>
  <em>Population of exoplanets as of 2017 (NASA/Ames Research Center/Natalie Batalha/Wendy Stenzel).</em>
</p>

There is currently a revolution ongoing in the field of exoplanets. New observatories have created unprecedented opportunities to detect and study these bodies in ways that were previously impossible. Missions such as the [Transiting Exoplanet Survey Satellite (TESS)](https://science.nasa.gov/mission/tess/) has measured transit signatures of thousands of potential exoplanets. The [James Webb Space Telescope (JWST)](https://science.nasa.gov/mission/webb/) allows us to measure the composition of the atmospheres of exoplanets. The [Atacama Large millimeter/submillimeter Array (ALMA)](https://www.almaobservatory.org/en/home/) observatory gives us a new view of protoplanetary disks, the sites of planet formation, and allows us to study the environments and results of ongoing planet formation. Together, these, and other, observatories have given us a wealth of data that will continue to provide discoveries for years to come.


<p align="center">
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/2/2a/JWST_spacecraft_model_3.png"
    width="500"
    alt="Model of the James Webb Space Telescope"
  /><br>
  <em>JWST (NASA).</em>
</p>


<p align="center">
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/The_Moon_and_the_Arc_of_the_Milky_Way01.jpg/1280px-The_Moon_and_the_Arc_of_the_Milky_Way01.jpg"
    width="750"
    alt="ALMA antennas beneath the Milky Way in the Atacama Desert"
  /><br>
  <em>ALMA (ESO/S. Guisard).</em>
</p>


Machine learning has been proven to be powerful tool in analyzing the trove of observation data. Different observatories create different types of data, including spectra, light curves, and images. Each type of data provides a different set of tasks, challenges, and information. This creates a situation in which a variety of machine learning techniques can be used for a broad set of analysis objectives. Because the ground truth of observations is often unknown, researches may rely on the creation of synthetic data through methods such as simulations to train models on known parameters before deployment on real datasets.

## Project
The purpose of EXXA is to both synthetic and observational data to perform many different tasks related to exoplanets. EXXA focuses on two main areas: exoplanet atmospheres and protoplanetary disks. There are different tasks for each area. The objectives of the atmosphere projects are mainly to identify chemical species to understand information such as the composition, weather, and potential habitability of planets. Protoplanetary disks are analyzed to identify planets, with intermediate goals including denoising the observations. 

### Previous results
The projects have resulted in, e.g., [publications](https://iopscience.iop.org/article/10.3847/1538-4357/aca477) and [conference talks](https://neurips.cc/virtual/2023/76151). Past GSoC projects have focused on

* [Equivarent model applied to disks (Alexandra Murariu)](https://medium.com/@murariu.alexandra2002/gsoc-ml4sci-exxa-equivariant-vision-networks-for-predicting-planetary-systems-architectures-b6f7c5846bda)

* [Exoplanet atmosphere characterization (Gaurav Shukla)](https://medium.com/@shuklag554/exoplanet-atmosphere-characterization-gsoc24-ml4sci-part-2-96392e3ba190)

* [Diffusion models to denoise disk observations (Faithful Chukwunwogor)](https://medium.com/@chukwuivnez/a-diffusion-based-deep-learning-framework-for-denoising-protoplanetary-disk-observations-gsoc-4870df837409)

* [Foundation models for general disk characterization (Tanmay Singhal)](https://medium.com/@singhaltanmay55/foundation-models-for-exoplanet-characterization-9ab7ef402f08)

Upcomming projects will expand on these results and may include additional capabilities, such as planet segmentation and simulation-based inference.

### Mentors
The mentors for these projects are

* [Katia Matcheva](mailto:ml4-sci@cern.ch) (University of Alabama)
* [Konstantin Matchev](mailto:ml4-sci@cern.ch) (University of Alabama)
* [Sergei Gleyzer](mailto:ml4-sci@cern.ch) (University of Alabama)
* [Jason Terry](mailto:jpterry@uga.edu) (Oxford University)
* [Alex Roman](mailto:ml4-sci@cern.ch) (University of Alabama)
* [Emilie Panek](mailto:ml4-sci@cern.ch) (University of Alabama)

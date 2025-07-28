###################################################################


Procedure to generate FARGO3D outputs for systems in this repo.

The folder `fargo_1_dust_multi_planet` contains a model setup for FARGO3D code

Process to run the FARGO3D simulations

    (1) First copy this folder to FARGO3D/public/setups 
        -  These setup specifications (Physics related flags, output specification flags etc.) can be found from fargo_1_dust_multi_planet.opt file within it.
        - It also contains various parameter files. There are in total 1000 such files, each of them specifies parameters for an individual protoplanetary disk system.
        - The parameter files are named as `fargo_1_dust_multi_planet_i.par`, here i denotes the system id.

    (2) Second step is to copy contains of `planet_files` folder to FARGO3D/public/planets
        - The `planet_files` folder contains planet configuration files corresponding to protoplanetary disk systems.
        - There are in total 1000 files
        - Filenames are of the form `system_i_planets.cfg`, where i is the integer denoting system's id.
        - Naturally, the `system_i_planets.cfg` file corresponds to `fargo_1_dust_multi_planet_i.par` file.

    (3) First we need to build the setup.
        - Run `make SETUP=fargo_1_dust_multi_planet PARALLEL=1` (with any additional flags needed for GPU)
        - Standard FARGO3D successful build message should come if build process goes es expected

    (4)  Run the simulations as usual by using commands
        - mpirun -np 2 ./fargo3d setups/fargo_1_dust_multi_planet/fargo_1_dust_multi_planet_4.par

        (here 2 represents number of threads used)


Use of utils.py to generate new parameter files and planet configuration files

(1) utils.py contains functions to (1) generate the parameter cube (using LHC sampling) for various systems, (2) Generate planet configuration files, (3) Generate corresponding system parameter files. 

The docstrings describes the parameters and functions usage. 

The function to generate system parameter files will use the default `fargo_1_dust_multi_planet.par` file as a templet and generate system parameter files
such as `fargo_1_dust_multi_planet_1.par`.
While generating system parameter files, it will replace the values of the LHC sampled parameters and all other parameters will remain similar to the templet parameter file.

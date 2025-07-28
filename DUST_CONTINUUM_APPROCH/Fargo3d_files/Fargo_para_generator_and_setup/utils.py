import numpy as np
from scipy.stats import qmc
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def lhc_sampler(params={}, number_of_samples=5, outfile=None):
    """
    Generates samples for parameters given by keys of the dictionary in given
    range of values.
    Returns parameter cube (stored in LHC.dat file).
    If save_cube = True then saves cube as .dat file at the location given in
    `outfile`.

    for e.g. 
    lhc_sampler(params = {'Sigmaslope':[0.5,1.0,'float'],'invstokes':[0.4,0.6,'float']},
    number_of_samples = 1000)
    will generate 1000 parameter pairs, the first colunmn of outputcube will contain 
    float values in range of 0.5 to 1.0 corresponding t Sigmaslope, and so on.
    The first row of the output file will contain column names.
    The second row will store the data type of each column.
    The datacube will be printed from the third row.
    """
    
    num = len(params)

    lower_bounds = np.array([value[0] for value in params.values()])
    upper_bounds = np.array([value[1] for value in params.values()])

    sampler = qmc.LatinHypercube(d=num, seed=7)
    sample = sampler.random(n=number_of_samples)
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

    # find out if any parameter needs to be an integer
    dtype_list = [value[2] for value in params.values()]
    scale = [value[3] for value in params.values()]

    indices = [index for index, element in enumerate(dtype_list) if element == 'int']
    for i in indices:
        sample_scaled[:,i] = np.round(sample_scaled[:,i]).astype('int')

    indices = [index for index, scale_type in enumerate(scale) if scale_type == 'log10']
    for i in indices:
        sample_scaled[:,i] = 10**(sample_scaled[:,i])  


    # Create a dtype for structured array
    dtype = np.dtype([(header, dtype) for header,dtype in zip(params.keys(), dtype_list)])

    # Create structured array
    structured_samples_cube = np.empty(sample_scaled.shape[0], dtype=dtype)
    for i, header in enumerate(params.keys()):
        structured_samples_cube[header] = sample_scaled[:, i]

    # Define format specifiers based on data type
    format_specifiers = {
        'int': '%d',
        'float': '%.3e'  # Using scientific notation with 3 decimal places
    }

    # Format the data for saving
    formatted_samples_cube = []
    for row in structured_samples_cube:
        formatted_row = []
        for header, dt in zip(params.keys(), dtype_list):
            formatted_row.append(format_specifiers[dt] % row[header])
        formatted_samples_cube.append(formatted_row)

    formatted_samples_cube = [dtype_list] + formatted_samples_cube
    
    # print(formatted_samples_cube)
    
    if outfile != None:
        if os.path.isfile(outfile):
            print("Old data file detected, deleted, making new")
            os.remove(outfile)
            np.savetxt(fname=outfile, X=formatted_samples_cube, delimiter='\t', fmt='%s', 
                       header='\t'.join(params.keys()), comments='')
        else:
            np.savetxt(fname=outfile, X=formatted_samples_cube, delimiter='\t', fmt='%s', 
                       header='\t'.join(params.keys()), comments='')



    return

lhc_sampler(params= parameters_dict,number_of_samples=1000,
            outfile = '/home/mihir/Fargo_data_generator/LHC.dat')


def planet_config_file_generator(parameters_file, columns_to_extract, output_file_path,
                                 feels_disk='YES', feels_others='YES', mass_range=(3e-6, 5e-3), 
                                 distance_range=(1.0, 4.0)):
    
    """
    Generates planet configuration files. 
    

    Parameters:
    - parameters_file (str)            : Path to the LHC.dat file.
    - columns_to_extract (list of str) : Which columns are needed to generate 
                                        planet configuration files from the LHC.dat file
    - feels_disk (str)                 : "YES" or "NO", if the planets feel diks's gravity or not. Default - "YES"
    - feels_others (str)               : "YES" or "NO", if the planets feel other planets' gravity or not. Default - "YES"
    - output_file_path (str)           : Path to directory where the generated planet config files 
                                         will be saved.
    - distance_range (tupple of floats): In case if system have more then one planet, the 
                                         distance of planets other then the first planet will 
                                         be randomly chosen from this range. It should be consistant
                                         with range specified in the lhc_sampler.
    - mass_range (tupple of floats)    : Same as distance range, mass range is supposed to be given 
                                         as log(mass range) - It will be stored in planet.cfg files 
                                         as mass. 
    
    """

    with open(parameters_file, 'r') as file:
        header_line = file.readline()
        data_types_line = file.readline()
    
    column_titles = header_line.strip().split('\t')
    data_types = data_types_line.strip().split('\t')
    
    # Load only the needed columns using the column titles and their data types
    usecols = [column_titles.index(col) for col in columns_to_extract]
    use_dtypes = [data_types[column_titles.index(col)] for col in columns_to_extract]
    data = np.loadtxt(parameters_file, delimiter='\t', dtype=str, usecols=usecols, skiprows=2)

    column_data = {}
    for col, dtype in zip(columns_to_extract, use_dtypes):
        idx = columns_to_extract.index(col)  # Get the index of the column
        # print(f"col {col}, idx = {idx}, dtype = {dtype}")
        if dtype == 'int':
            column_data[col] = data[:, idx].astype(int)
        elif dtype == 'float':
            column_data[col] = data[:, idx].astype(float)
        else:
            column_data[col] = data[:, idx]

    pmass_distribution = []
    
    if 'planets' in column_titles and 'planet_distance' in column_data and 'planet_mass' in column_data:
        for i, (num_planets, planet_dist, planet_mass) in enumerate(zip(column_data['planets'],
                                                                         column_data['planet_distance'],
                                                                         column_data['planet_mass'])):
            output_file_name = f"system_{i}_planets.cfg"
            output_file_path_i = os.path.join(output_file_path, output_file_name)
            
            with open(output_file_path_i, 'w') as output_file:
                output_file.write("# PlanetName  Distance  Mass  Accretion  Feels Disk  Feels Others\n")
                
                num_planets = int(num_planets)  # Convert to integer
                
                if num_planets == 0:
                    planet_mass = 0.0
                    output_line = f"planet_0  {planet_dist:.3e}  {planet_mass:.3e}  0.0  {feels_disk}  {feels_others}\n"
                    output_file.write(output_line)
                elif num_planets == 1:
                    output_line = f"planet_0  {planet_dist:.3e}  {planet_mass:.3e}  0.0  {feels_disk}  {feels_others}\n"
                    output_file.write(output_line)
                else:
                    for planet_id in range(num_planets):
                        if planet_id == 0:
                            planet_mass = column_data['planet_mass'][i]
                            planet_dist = column_data['planet_distance'][i]
                        else:
                            planet_mass = np.exp(np.random.uniform(*mass_range))
                            planet_dist = np.random.uniform(*distance_range) 
                            pmass_distribution.append(planet_mass)                       
                        output_line = f"planet_{planet_id}  {planet_dist:.3e}  {planet_mass:.3e}  0.0  {feels_disk}  {feels_others}\n"
                        output_file.write(output_line)   
    else:
        print("No columns specifying planet numbers found")
    
    return pmass_distribution


planets_mass = planet_config_file_generator(parameters_file="Fargo_data_generator/LHC.dat", 
                            columns_to_extract = ['planets',
                                                     'planet_distance',
                                                     'planet_mass'], 
                              output_file_path = "/home/mihir/Fargo_data_generator/planet_files",
                              feels_disk='YES', feels_others='YES', 
                              mass_range=(np.log(3.0e-6),np.log(5.0e-3)), distance_range=(1.0, 8.5))


def parameter_file_generator(fargo_par_file="Fargo_data_generator/fargo_1_dust_multi_planet/fargo_1_dust_multi_planet.par",
                             LHC_parameter_file="Fargo_data_generator/LHC.dat",
                             planet_files_dir="Fargo_data_generator/planet_files",
                             output_dir="Fargo_data_generator/Fargo_system_par_files"):

    """

    Generates parameter files based on a template FARGO parameter file and LHC data.

    Parameters:
    - fargo_par_file (str): Path to the template FARGO parameter file.
    - LHC_parameter_file (str): Path to the LHC parameter file containing planetary system data.
    - planet_files_dir (str): Directory containing planet configuration files.
    - output_dir (str): Directory where the generated parameter files will be saved.


    
    This function reads a template FARGO parameter file and an LHC parameter file. The template
    file should have one parameter per line, using whitespace separation between the parameter name
    and its value. The LHC parameter file contains a header row with parameter names, followed by
    a second row specifying the data types. The subsequent rows contain Latin hypercube sampled data.


    Reads the fargo parameter file as a dictionary, This file is going to be used as a tempelet, 
    Values of parameters listed in the LHC_parameter file which are present in the fargo_par_file
    will be changed. Other parameters will be left untouched.
    
    It loops through rows in LHC_parameter_file rows (each row correspond to parameters of a perticular system),
    Saves fargo parameter files for each system in form of filename of 'fargo_par_file_i.par' where i is system number.
    Saves this file in output_dir.

    Additionally it changes the value of planet configuration file "PlanetConfig" parameter
    Looks for planet file of the name `system_i_planets.cfg" in planet_files_dir. (Here "i" is system number.)
    If it exist then changes "PlanetConfig" value to the corresponding planet sysstem file.

    Calculates the time interval DT and Ninterm such that DT*Ninterm is 1/4*(orbital period of the outermost planet)
    And then sets Ntot = 50*Ninterm (total number of outputs needs to be 50)

    Finally changes the value of OutputDir to @outputs/fargo_1_dust_multi_planet_i where i is the system number.


    For each row in the LHC data:
    Checks if a corresponding planet configuration file (system_i_planets.cfg) exists in planet_files_dir
        1. If it exists the "PlanetConfig" parameter is updated with the path of the system_i_planets.cfg file in planets directory.
        2. The largest orbit distance for planets in system is determined, and based on that
            time period for coarse grain output (1/4 th of orbital period of largest planet)
            is determined and total evolution time is determined such that we get total 50 coarse grain outputs
        3. System parameters listed in LHC_parameter_files and present in FARGO3D parameter tempelet file are updated
           with values listed in LHC_parameter_file.
        4. The "OutputDir" parameter is updated based on the system id (row index) and filename of the tempelet parameter file.
        5. Creates a new parameter file using the updated dictionary and saved in the output directory.
        6. If same named parameter file already exist in the output directory then overwrites it.

    Note:
    - The template FARGO parameter file (templete) should be structured with one parameter per line,
      using whitespace separation between the parameter name and its value.
    - The LHC parameter file should contain a header row with parameter names, followed by
      a second row specifying the data types. The subsequent rows contain the experimental data.

    Example usage:
    parameter_file_generator()

    """
    # Initialize an empty dictionary to store parameter values
    parameter_dict = {}

    # Read the fargo_par_file line by line and store parameters as a dictionary
    with open(fargo_par_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            words = line.split()
            if len(words) >= 2:
                key = words[0]
                value = words[1]
                parameter_dict[key] = value

    # Read LHC_parameter_file as a Pandas DataFrame
    column_names = []
    data_types = []
    with open(LHC_parameter_file, 'r') as file:
        # Read the column names line
        column_names = file.readline().split()
        # Read the data types line
        data_types = file.readline().split()

    # Load the data using Pandas, skipping the first two rows
    data = pd.read_csv(LHC_parameter_file, skiprows=2, sep='\t', names=column_names, dtype=dict(zip(column_names, data_types)))

    # Get the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(fargo_par_file))[0]

    # Iterate through the rows of the DataFrame
    for index, row in data.iterrows():

        # Check if "system_i_planets.cfg" file exists in planet_files_dir
        planet_config_file = f"{planet_files_dir}/system_{index}_planets.cfg"
        if os.path.exists(planet_config_file):
            try:
                # Read the second column (Distance) from the file using np.loadtxt
                planet_distances = np.loadtxt(planet_config_file, skiprows=1, usecols=1)
                largest_orbit_distance = np.max(planet_distances)
            except ValueError:
                # If there's only one value, it will raise a ValueError
                planet_distances = np.loadtxt(planet_config_file, skiprows=1, usecols=1)
                largest_orbit_distance = planet_distances            
            # Calculate the max orbital period
            max_orbital_period = 2 * np.pi * np.sqrt(largest_orbit_distance ** 3.)
            
            # Calculate ninterm
            ninterm = int(max_orbital_period / (4 * float(parameter_dict["DT"])))

            # Update Ninterm and Ntot in the parameter_dict
            parameter_dict["Ninterm"] = str(ninterm)
            parameter_dict["Ntot"] = str(50 * ninterm)
            
            # Update the values in the parameter_dict based on the data in LHC
            for column in column_names:
                if column in parameter_dict:
                    parameter_dict[column] = row[column]
                    # print(f"col name is {column}, value is {parameter_dict[column]}")
            
            # Update value of PlanetConfig parameter by saving corresponding file name in it
            parameter_dict["PlanetConfig"] = f"planets/system_{index}_planets.cfg"
            # Update the OutputDir key
            parameter_dict["OutputDir"] = f"@outputs/{filename_without_extension}_{index}"
            
            # Create a parameter file for each row using Pandas to_csv
            output_file_name = f"{output_dir}/{filename_without_extension}_{index}.par"
            output_df = pd.DataFrame(parameter_dict.items(), columns=['Key', 'Value'])
            output_df.to_csv(output_file_name, sep='\t', index=False)
        else:
            # Print an error message and raise an exception
            error_message = f"'system_{index}_planets.cfg' file not found in '{planet_files_dir}'"
            raise FileNotFoundError(error_message)
        
# Example usage
parameter_file_generator()






'''
TODO: In another script the data files need to be checked

as well as the data files directories hierarchy

Effective_mass_squared_Vs_bare_mass_values
Time_dependence_of_effective_mass_values
Time_dependence_of_jackknife_averaged_momentum_correlator

No empty data files subdirectories

Don't forget the equal signs like KL=1, or N=20

Data filenames must contain all the operator type info!

'''

import os
import sys
import re

import numpy as np
import matplotlib.pyplot as plt
import click
import gvar as gv
import lsqfit


sys.path.append('../')
import custom_library.directories_organization as directories_organization
import custom_library.info_extraction as info_extraction
import custom_library.jackknife_analysis as jackknife_analysis
import custom_library.effective_mass as effective_mass
import custom_library.momentum_correlator as momentum_correlator
import custom_library.fit_functions as fit_functions


@click.command()
@click.option("--data_files_directory", "data_files_directory", "-data_dir", default='../data_files/', help="N/A")
@click.option("--data_files_extension", "data_files_extension", "-data_ext", default='.dat', help="N/A")
@click.option("--plotting_directory", "plotting_directory", "-plot_dir", default='../plots/', help="N/A")


def main(data_files_directory, data_files_extension, plotting_directory):

    # PREPARE THE SUBDIRECTORIES STRUCTURE OF THE PLOTTING DIRECTORY

    # Replicate the subdirectories structure of the data files directory inside the plotting directory up to depth 2. Ignore if it is already established. Store in a list all the created plotting subdirectories without any filter
    list_of_created_subdirectories = directories_organization.replicate_directory_structure(data_files_directory, plotting_directory)
    # Select only those subdirectories that have leaf subdirectories "Standard" or "Brillouin"
    plotting_subdirectories_paths_list = [subdirectory for subdirectory in list_of_created_subdirectories if any(substring in subdirectory for substring in ["Standard", "Brillouin"])]

    # Additionally create in each subdirectory another two subdirectories from the following list for storing plots per kappa value
    additional_plotting_subdirectories_names_list = ["Time_dependence_of_effective_mass_values", "Time_dependence_of_jackknife_averaged_momentum_correlator"]
    for plotting_subdirectory_path in plotting_subdirectories_paths_list:
        for additional_plotting_subdirectory in additional_plotting_subdirectories_names_list:
            additional_plotting_subdirectory_path = os.path.join(plotting_subdirectory_path, additional_plotting_subdirectory)
            if not os.path.exists(additional_plotting_subdirectory_path):
                os.makedirs(additional_plotting_subdirectory_path)

    # List the paths of all subdirectories of the input data files directory that contain dataset files
    data_files_subdirectories_paths_list = directories_organization.list_leaf_subdirectories(data_files_directory)
    # Filter those that contain the "/processed_data_files" leaf subdirectory
    processed_data_files_subdirectories_paths_list = [subdirectory for subdirectory in data_files_subdirectories_paths_list if "/processed_data_files" in subdirectory]

    critical_mass_values_dictionary = dict()
    for processed_data_files_subdirectory_path in processed_data_files_subdirectories_paths_list:

        # For later use
        # Extract the basename of the file or subdirectory
        operator_type_label = os.path.basename(processed_data_files_subdirectory_path)

        # Extract operator info from subdirectory name
        operator_type_info = info_extraction.operator_type_extraction(operator_type_label)

        operator_type, operator_method, operator_method_enumeration = operator_type_info

        print(f'\nWorking with {os.path.basename(processed_data_files_subdirectory_path)} datasets:')

        # ANALYZE EACH DATA FILE IN THE SUBDIRECTORY

        # Initialize dictionary with kappa values as keys and nested dictionaries as values which in their turn have configuration labels as keys and for values a NumPy array with the momentum correlator values
        momentum_correlator_values_per_kappa_per_configuration_dictionary = dict()
        for data_file_path in os.listdir(processed_data_files_subdirectory_path):
            # Additional safety check if data file ends with appropriate extension
            if data_file_path.endswith(data_files_extension):

                # Properties of each dataset are extracted from its filename. Its content are the values of the momentum correlator. Internally an additional check is performed on operator type info
                data_file_path = os.path.join(processed_data_files_subdirectory_path, data_file_path)
                dataset_object = info_extraction.AnalyzeDataset(data_file_path, operator_type_info)
                
                # If the list of keys does not contain the kappa value then initialize the corresponding value with a nested dictionary
                if dataset_object.kappa not in momentum_correlator_values_per_kappa_per_configuration_dictionary:
                    momentum_correlator_values_per_kappa_per_configuration_dictionary[dataset_object.kappa] = dict()

                # Pass the dataset object as value
                momentum_correlator_values_per_kappa_per_configuration_dictionary[dataset_object.kappa][dataset_object.configuration_label] = dataset_object
        
        # WORK WITH DATASETS OF THE SAME KAPPA VALUE

        effective_mass_replica_estimates_per_kappa_value_dictionary = dict()
        for kappa_value in momentum_correlator_values_per_kappa_per_configuration_dictionary.keys():

            print(f'* for κ={kappa_value}...')

            # Create a 2D NumPy array with all the 1D NumPy arrays with zero-momentum correlator values of the same kappa value
            original_momentum_correlator_values_per_kappa_2D_array = np.array([dataset_object.contents_array for dataset_object in momentum_correlator_values_per_kappa_per_configuration_dictionary[kappa_value].values()])

            # Assuming all time-dependent momentum correlator arrays have the same length?????????? for later use
            temporal_direction_lattice_size = original_momentum_correlator_values_per_kappa_2D_array.shape[1]

            # This 2D array will be used in the jackknife analysis                 
            jackknife_analyzed_momentum_correlator_values_per_configuration_object = jackknife_analysis.JackknifeAnalysis(original_momentum_correlator_values_per_kappa_2D_array)

            # JACKKNIFE AVERAGED TIME-DEPENDENT ZERO-MOMENTUM CORRELATOR VALUES

            jackknife_average_of_momentum_correlator = jackknife_analyzed_momentum_correlator_values_per_configuration_object.jackknife_average

            # Symmetrization about the t = T//2 axis is implemented for reducing the error of each point of the jackknife-averaged momentum correlator
            jackknife_average_of_momentum_correlator = momentum_correlator.symmetrization(jackknife_average_of_momentum_correlator)

            # OPTIMUM FITTING RANGES

            # Calculate the time-dependent values of effective mass from the jackknife-averaged time-dependent momentum correlator. This array has half the length of the input array.
            time_dependent_effective_mass_values_array = effective_mass.effective_mass_periodic_case_function(jackknife_average_of_momentum_correlator)

            # For the plateau fit (single) parameter starting assign the value of the time-dependent effective mass array values at half-its-length position. This choice is arbitrary, but is more likely to be close to the best-fit estimate while avoiding any extreme values at the edges of the range
            effective_mass_plateau_fit_p0 = [gv.mean(time_dependent_effective_mass_values_array[dataset_object.temporal_direction_lattice_size//4])]
            # Since the t=0 value is excluded from the  time-dependent effective mass array, then the corresponding times range is shifted by +1
            y = time_dependent_effective_mass_values_array
            x = range(1, len(y)+1)
            effective_mass_plateau_fit_optimum_range, effective_mass_optimum_plateau_best_fit = effective_mass.optimum_range(x, y, effective_mass.plateau_fit_function, effective_mass_plateau_fit_p0)

            # Initialize two-state fit parameters ?????
            effective_mass_two_state_fit_p0 = [gv.mean(effective_mass_optimum_plateau_best_fit.p[0]), 10, 1]
            effective_mass_two_state_fit_optimum_range, effective_mass_optimum_two_state_best_fit = effective_mass.optimum_range(x, y, effective_mass.two_state_fit_function, effective_mass_two_state_fit_p0)

            # PLOT TIME DEPENDENCE OF EFFECTIVE MASS VALUES

            fig, ax = plt.subplots()
            ax.grid()
            ax.set_title('Time dependence of effective mass values\nfrom jackknife-averaged zero-momentum correlator values', pad = 10)
            ax.set(xlabel='$t$', ylabel='$\\log\\left( \\frac{C(t-1)+\\sqrt{C^2(t-1)-C^2(T//2)})}{C(t+1)+\\sqrt{C^2(t+1)-C^2(T//2)})} \\right) $')
            plt.subplots_adjust(left=0.15) # Adjust left margin
            ax.set_xlim(0, temporal_direction_lattice_size//2)

            plt.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10, label=f'{operator_type_label}, $\\kappa$={kappa_value:.3f}, T={temporal_direction_lattice_size},\n# of configs={jackknife_analyzed_momentum_correlator_values_per_configuration_object.sample_size}, APE iters=1, cw=1')

            # Plateau fit
            x_data = np.linspace(x[effective_mass_plateau_fit_optimum_range[0]], x[effective_mass_plateau_fit_optimum_range[1]-1], 100)
            y_data = effective_mass.plateau_fit_function(x_data, gv.mean(effective_mass_optimum_plateau_best_fit.p))
            plt.plot(x_data, y_data, 'r--', label=f'Optimum plateau fit: $m^{{best_fit}}_{{eff.}}$={effective_mass_optimum_plateau_best_fit.p[0]:.3f}')
                    #  \n($\\chi^2$/dof={effective_mass_optimum_plateau_best_fit.chi2:.2f}/{effective_mass_optimum_plateau_best_fit.dof}={effective_mass_optimum_plateau_best_fit.chi2/effective_mass_optimum_plateau_best_fit.dof:.2f}), p-value={1-effective_mass_optimum_plateau_best_fit.Q:.2e})')???????????
            
            # Two-state fit
            x_data = np.linspace(x[effective_mass_two_state_fit_optimum_range[0]], x[effective_mass_two_state_fit_optimum_range[1]-1], 100)
            plt.plot(x_data, effective_mass.two_state_fit_function(x_data, gv.mean(effective_mass_optimum_two_state_best_fit.p)), 'g--', label=f'Optimum two-state fit: $m^{{best_fit}}_{{eff.}}$={effective_mass_optimum_two_state_best_fit.p[0]:.3f}')
                    #  \n($\\chi^2$/dof={effective_mass_optimum_two_state_best_fit.chi2:.2f}/{effective_mass_optimum_two_state_best_fit.dof}={effective_mass_optimum_two_state_best_fit.chi2/effective_mass_optimum_two_state_best_fit.dof:.2f}), p-value={1-effective_mass_optimum_two_state_best_fit.Q:.2e})')???????????

            ax.legend(loc="upper center")

            # ?????????
            plotting_subdirectory = directories_organization.change_root(processed_data_files_subdirectory_path, data_files_directory, plotting_directory)
            effective_mass_plotting_subdirectory = os.path.join(plotting_subdirectory, additional_plotting_subdirectories_names_list[0])
            # Additional check
            if not os.path.exists(effective_mass_plotting_subdirectory):
                raise ValueError(f'No appropriate "{additional_plotting_subdirectories_names_list[0]}" subdirector was created.')

            plot_filename = f'Time_dependence_of_effective_mass_values_{operator_type_label}_kappa={kappa_value}.png'
            plot_path = os.path.join(effective_mass_plotting_subdirectory, plot_filename)

            print("Effective mass plot created.")

            fig.savefig(plot_path)
            plt.close()

            # PLOT TIME DEPENDENCE OF THE JACKKNIFE-AVERAGED ZERO-MOMENTUM CORRELATOR VALUES

            fig, ax = plt.subplots()
            ax.grid()
            ax.set_title('Time dependence of the jackknife-averaged\nzero-momentum correlator values', pad = 10)
            ax.set(xlabel='$t$', ylabel='$C(t)$')
            ax.set_yscale('log')
            # plt.subplots_adjust(left=0.15) # Adjust left margin
            # ax.set_xlim(0, temporal_direction_lattice_size//2)

            x = np.array(range(temporal_direction_lattice_size))
            y = jackknife_average_of_momentum_correlator

            plt.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10, label=f'{operator_type_label}, $\\kappa$={kappa_value:.3f}, T={temporal_direction_lattice_size},\n# of configs={jackknife_analyzed_momentum_correlator_values_per_configuration_object.sample_size}, APE iters=1, cw=1')

            # Plateau fit
            # Use the value???????????????
            effective_mass_estimate = gv.mean(effective_mass_optimum_plateau_best_fit.p[0])
            amplitude_factor = gv.mean(y[temporal_direction_lattice_size//4])/(np.exp(-effective_mass_estimate*(temporal_direction_lattice_size//4)) + np.exp(-effective_mass_estimate*(3*temporal_direction_lattice_size//4)))
            momentum_correlator_plateau_fit_p0 = [amplitude_factor, effective_mass_estimate]
            momentum_correlator_plateau_fit = lsqfit.nonlinear_fit(data=(x[effective_mass_plateau_fit_optimum_range[0]:effective_mass_plateau_fit_optimum_range[1]], y[effective_mass_plateau_fit_optimum_range[0]:effective_mass_plateau_fit_optimum_range[1]]), p0=momentum_correlator_plateau_fit_p0, fcn=momentum_correlator.plateau_periodic_correlator, debug=True)

            x_data = np.linspace(min(x), max(x), 100)
            plt.plot(x_data, momentum_correlator.plateau_periodic_correlator(x_data, gv.mean(momentum_correlator_plateau_fit.p)), 'r--', label=f'Plateau fit: $m^{{best_fit}}_{{eff.}}$={momentum_correlator_plateau_fit.p[1]:.3f}\n($\\chi^2$/dof={momentum_correlator_plateau_fit.chi2:.2f}/{momentum_correlator_plateau_fit.dof}={momentum_correlator_plateau_fit.chi2/momentum_correlator_plateau_fit.dof:.2f}), p-value={1-momentum_correlator_plateau_fit.Q:.2e})')
            
            # # Two-state fit
            # Pass
            amplitude_factor = gv.mean(momentum_correlator_plateau_fit.p[0])
            effective_mass_estimate = gv.mean(effective_mass_optimum_two_state_best_fit.p[0])
            r = gv.mean(effective_mass_optimum_two_state_best_fit.p[1])
            c = gv.mean(effective_mass_optimum_two_state_best_fit.p[2])
            momentum_correlator_two_state_fit_p0 = [amplitude_factor, effective_mass_estimate, r, c]
            momentum_correlator_two_state_fit = lsqfit.nonlinear_fit(data=(x[effective_mass_two_state_fit_optimum_range[0]:effective_mass_two_state_fit_optimum_range[1]], y[effective_mass_two_state_fit_optimum_range[0]:effective_mass_two_state_fit_optimum_range[1]]), p0=momentum_correlator_two_state_fit_p0, fcn=momentum_correlator.two_state_periodic_correlator, debug=True)

            plt.plot(x_data, momentum_correlator.two_state_periodic_correlator(x_data, gv.mean(momentum_correlator_two_state_fit.p)), 'g--', label=f'Two-state fit: $m^{{best_fit}}_{{eff.}}$={momentum_correlator_two_state_fit.p[1]:.3f}\n($\\chi^2$/dof={momentum_correlator_two_state_fit.chi2:.2f}/{momentum_correlator_two_state_fit.dof}={momentum_correlator_two_state_fit.chi2/momentum_correlator_two_state_fit.dof:.2f}), p-value={1-momentum_correlator_two_state_fit.Q:.2e})')

            ax.legend(loc="upper center")

            momentum_correlator_plotting_subdirectory = os.path.join(plotting_subdirectory, additional_plotting_subdirectories_names_list[1])
            # Additional check
            if not os.path.exists(momentum_correlator_plotting_subdirectory):
                raise ValueError(f'No appropriate "{additional_plotting_subdirectories_names_list[1]}" subdirector was created.')

            plot_filename = f'Time_dependence_of_jackknife_averaged_momentum_correlator_{operator_type_label}_kappa={kappa_value}.png'
            
            plot_path = os.path.join(momentum_correlator_plotting_subdirectory, plot_filename)

            print("Momentum correlator plot created.")

            fig.savefig(plot_path)
            plt.close()

            # EFFECTIVE MASS ESTIMATE PER CONFIGURATION ???

            effective_mass_replica_estimates_list = list()
            for jackknife_replica_of_momentum_correlator_values in jackknife_analyzed_momentum_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array:
                jackknife_replica_of_momentum_correlator_values = momentum_correlator.symmetrization(jackknife_replica_of_momentum_correlator_values)

                jackknife_replica_of_momentum_correlator_values = gv.gvar(jackknife_replica_of_momentum_correlator_values, gv.sdev(jackknife_average_of_momentum_correlator))

                replica_effective_mass_values_array = effective_mass.effective_mass_periodic_case_function(jackknife_replica_of_momentum_correlator_values)

                y = replica_effective_mass_values_array
                x = range(1, len(y)+1)

                effective_mass_plateau_fit_p0 = [gv.mean(effective_mass_optimum_plateau_best_fit.p[0])]
                replica_effective_mass_plateau_fit = lsqfit.nonlinear_fit(data=(x[effective_mass_plateau_fit_optimum_range[0]:effective_mass_plateau_fit_optimum_range[1]], y[effective_mass_plateau_fit_optimum_range[0]:effective_mass_plateau_fit_optimum_range[1]]), p0=effective_mass_plateau_fit_p0, fcn=effective_mass.plateau_fit_function, debug=True)

                effective_mass_replica_estimates_list.append(gv.mean(replica_effective_mass_plateau_fit.p[0]))

            effective_mass_replica_estimates_per_kappa_value_dictionary[kappa_value] = effective_mass_replica_estimates_list

        # PLOT EFFECTIVE MASS SQUARED AGAINST BARE MASS VALUES

        # Calculate ????????????????????????
        kappa_values_array = np.array(list(effective_mass_replica_estimates_per_kappa_value_dictionary.keys()))
        bare_mass_values_array = 0.5/kappa_values_array - 4.

        # Extract values from the dictionary and stack them into a NumPy array
        effective_mass_replica_estimates_2D_array = np.vstack(list(effective_mass_replica_estimates_per_kappa_value_dictionary.values()))
        effective_mass_replica_estimates_2D_array_squared = np.square(effective_mass_replica_estimates_2D_array)
        # Compute the covariance matrix
        effective_mass_replica_estimates_covariance_matrix = np.cov(effective_mass_replica_estimates_2D_array_squared, rowvar=True, ddof=0)*(effective_mass_replica_estimates_2D_array.shape[0]-1)

        average = np.average(effective_mass_replica_estimates_2D_array_squared, axis=1)
        error = np.std(effective_mass_replica_estimates_2D_array_squared, ddof=0, axis=1)*np.sqrt(effective_mass_replica_estimates_2D_array.shape[1]-1)

        average_squared_effective_mass_estimates_array = gv.gvar(average, error)

        # PLOT EFFECTIVE MASS SQUARED AGAIN BARE MASS VALUES

        fig, ax = plt.subplots()
        ax.grid()

        print(f"shape={effective_mass_replica_estimates_2D_array.shape}")

        sample_size = effective_mass_replica_estimates_2D_array.shape[1]
        ax.set_title(f'Effective pion mass squared Vs. bare mass values\n({operator_type_label}, # of configs={sample_size}, T={temporal_direction_lattice_size}, APE iters=1, cw=1)')
        ax.set(xlabel='$a m_b$', ylabel='$m^2_{eff.}$')

        ax.axhline(0, color='black') # y = 0
        ax.axvline(0, color='black') # x = 0

        # Sort
        sorted_indices = np.argsort(bare_mass_values_array)
        bare_mass_values_array = bare_mass_values_array[sorted_indices]
        average_squared_effective_mass_estimates_array = average_squared_effective_mass_estimates_array[sorted_indices]

        plt.errorbar(bare_mass_values_array, gv.mean(average_squared_effective_mass_estimates_array), yerr=gv.sdev(average_squared_effective_mass_estimates_array), fmt='.', markersize=8, capsize=10)
                    #  , label=f'Optimum fitting range: {critical_mass_optimum_range[0], critical_mass_optimum_range[1]-1}')

        # Find minimum and shift
        min_index = np.argmin(average_squared_effective_mass_estimates_array)
        if (min_index>0):
            bare_mass_values_array = bare_mass_values_array[min_index:]
            average_squared_effective_mass_estimates_array = average_squared_effective_mass_estimates_array[min_index:]

        # Investigate optimum range
        critical_mass_optimum_range = fit_functions.critical_mass_optimum_range(bare_mass_values_array, average_squared_effective_mass_estimates_array, sample_size)

        print(f'critical_mass_optimum_range={critical_mass_optimum_range}')

        # Linear fit
        x = bare_mass_values_array[critical_mass_optimum_range[0]:critical_mass_optimum_range[1]]
        y = average_squared_effective_mass_estimates_array[critical_mass_optimum_range[0]:critical_mass_optimum_range[1]]
        # # The initial estimate for the effective mass equals the value 
        slope = (max(gv.mean(y)) - min(gv.mean(y)))/(max(gv.mean(x)) - min(gv.mean(x)))
        linear_fit_p0 = [slope, -min(gv.mean(y))/slope+min(x)]
        index_cut = -1
        # [:index_cut]
        linear_fit = lsqfit.nonlinear_fit(data=(x, y), p0=linear_fit_p0, fcn=fit_functions.linear_function, debug=True)

        margin = 0.03
        if (gv.mean(linear_fit.p[1])<0):
            margin = -margin
        left_edge = linear_fit.p[1]
        if (operator_method == "KL"):
            left_edge = 0
        x_data = np.linspace(gv.mean(left_edge)*(1-margin), max(gv.mean(x))*(1+margin), 100)
        y_data = fit_functions.linear_function(x_data, linear_fit.p)
        kappa_critical = 0.5/(linear_fit.p[1]+4)
        plt.plot(x_data, gv.mean(y_data), 'r--', label=f'linear fit ($\\chi^2$/dof={linear_fit.chi2:.2f}/{linear_fit.dof}={linear_fit.chi2/linear_fit.dof:.2f}):\n$a m_c$={linear_fit.p[1]:.5f}, $\\kappa_c$={kappa_critical:.6f}')
        ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='r', alpha=0.2)

        # Quadratic fit
        x = bare_mass_values_array
        y = average_squared_effective_mass_estimates_array
        quadratic_fit_p0 = [gv.mean(linear_fit.p[0]), gv.mean(linear_fit.p[1]), 0.1*gv.mean(linear_fit.p[1])]
        quadratic_fit = lsqfit.nonlinear_fit(data=(x, y), p0=quadratic_fit_p0, fcn=fit_functions.quadratic_function, debug=True)
        left_edge = quadratic_fit.p[1]
        if (operator_method == "KL"):
            left_edge = 0
        x_data = np.linspace(gv.mean(left_edge)*(1-margin), max(gv.mean(x))*(1+margin), 100)
        y_data = fit_functions.quadratic_function(x_data, quadratic_fit.p)
        kappa_critical = 0.5/(quadratic_fit.p[1]+4)
        plt.plot(x_data, gv.mean(y_data), 'g--', label=f'quadratic fit ($\\chi^2$/dof={quadratic_fit.chi2:.2f}/{quadratic_fit.dof}={quadratic_fit.chi2/quadratic_fit.dof:.2f}):\n$a m_c$={quadratic_fit.p[1]:.5f}, $\\kappa_c$={kappa_critical:.6f}')
        ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='g', alpha=0.2)

        # Change legend's position
        if (gv.mean(linear_fit.p[1])>0):
            ax.legend(loc="upper left")
        else:
            ax.legend(loc="lower right")

        # ???
        y_axis_margin = 0.25
        ax.set_ylim(-max(gv.mean(average_squared_effective_mass_estimates_array))/3, max(gv.mean(average_squared_effective_mass_estimates_array)*(1 + y_axis_margin)))

        plot_filename = f'Effective_mass_squared_Vs_bare_mass_{operator_type_label}.png'
        plot_path = os.path.join(plotting_subdirectory, plot_filename)
        fig.savefig(plot_path)
        plt.close()

        print("Effective against bare mass plot created.")

        if ("Standard_Chebyshev" in operator_type_label):
            # or ("KL" in operator_type_label)
            critical_mass_values_dictionary[operator_type_label] = linear_fit.p[1], sample_size

    # PLOT CRITICAL MASS AGAINST OPERATOR METHOD ENUMERATION

    number_of_Chebyshev_terms_array = list()
    for operator_type_label in critical_mass_values_dictionary.keys():
        operator_type, operator_method, operator_method_enumeration = info_extraction.operator_type_extraction(operator_type_label)
        number_of_Chebyshev_terms_array.append(operator_method_enumeration)

    number_of_Chebyshev_terms_array = np.array(number_of_Chebyshev_terms_array)
    critical_mass_values_array = np.array([value_tuple[0] for value_tuple in critical_mass_values_dictionary.values()])
    sample_size_array = np.array([value_tuple[1] for value_tuple in critical_mass_values_dictionary.values()])

    print(number_of_Chebyshev_terms_array)
    print(critical_mass_values_array)

    sorted_indices = np.argsort(number_of_Chebyshev_terms_array)
    number_of_Chebyshev_terms_array = number_of_Chebyshev_terms_array[sorted_indices]
    critical_mass_values_array = critical_mass_values_array[sorted_indices]
    sample_size_array = sample_size_array[sorted_indices]

    fig, ax = plt.subplots()
    ax.grid()

    ax.set_title(f'Critical mass Vs. number of Chebyshev term\n({operator_type} {operator_method}, T={temporal_direction_lattice_size}, APE iters=1, cw=1)')
    ax.set(xlabel='N', ylabel='$a m^{critical}_b$')
    ax.set_yscale('log')
    ax.set_xlim(min(number_of_Chebyshev_terms_array) - 5, max(number_of_Chebyshev_terms_array) + 10)

    plt.errorbar(number_of_Chebyshev_terms_array, gv.mean(critical_mass_values_array), yerr=gv.sdev(critical_mass_values_array), fmt='.', markersize=8, capsize=10)

    # Exponential fit
    index_cut = 1
    x = number_of_Chebyshev_terms_array[index_cut:-2]
    y = critical_mass_values_array[index_cut:-2]

    attenuation_rate = gv.mean(-np.log(y[-1]/y[-2])/(x[-1]-x)[-2])
    amplitude_factor = gv.mean( y[-1]/np.exp(-attenuation_rate*x[-1]) )
    critical_mass_exponential_fit_p0 = [amplitude_factor, attenuation_rate]
    critical_mass_exponential_fit = lsqfit.nonlinear_fit(data=(x,y), p0=critical_mass_exponential_fit_p0, fcn=fit_functions.simple_exponential_function, debug=True)

    # Plot line
    x_data = np.linspace(min(number_of_Chebyshev_terms_array), max(number_of_Chebyshev_terms_array), 100)
    y_data = gv.mean(fit_functions.simple_exponential_function(x_data, critical_mass_exponential_fit.p))                                     

    plt.plot(x_data, y_data, 'r--', label=f'Exponential fit: {critical_mass_exponential_fit.p[0]:.2f}$\\times$exp(-{critical_mass_exponential_fit.p[1]:.2f}$\\times$N)')
    
    for index, sample_size in enumerate(sample_size_array):
        ax.annotate(sample_size, (number_of_Chebyshev_terms_array[index], gv.mean(critical_mass_values_array[index])), xytext=(0, 15), textcoords="offset pixels", color='brown', bbox=dict(facecolor='none', edgecolor='black'))

    ax.legend(loc="upper right")

    plotting_subdirectory = os.path.join(plotting_directory, operator_type+'_'+operator_method)
    plot_filename = f'Critical_bare_mass_against_number_Chebyshev_terms_{ operator_type}_{operator_method}.png'
    plot_path = os.path.join(plotting_subdirectory, plot_filename)
    fig.savefig(plot_path)
    plt.close()

    print("Critical mass against operator enumeration plot created.")


if __name__ == "__main__":

    main()

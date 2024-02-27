import numpy as np
import scipy.stats as stats
import lsqfit
import warnings


warnings.filterwarnings('ignore')


def effective_mass_periodic_case_function(array):

	lattice_size = np.shape(array)[0]
	middle_value_array = np.min(array)
    #  + (-1E-40)

    # lattice_size//2-2

	shifted_backward_array = np.roll(array, shift=+1)
	shifted_backward_array = shifted_backward_array[1:lattice_size//2-1]
	shifted_forward_array = np.roll(array, shift=-1)
	shifted_forward_array = shifted_forward_array[1:lattice_size//2-1]

	numerator = shifted_backward_array + np.sqrt(np.square(shifted_backward_array) - middle_value_array**2)
	denominator = shifted_forward_array + np.sqrt(np.square(shifted_forward_array) - middle_value_array**2)

	return 0.5*np.log(numerator/denominator)


def plateau_fit_function(x, p):

    return np.full(len(x), p)


def two_state_fit_function(x, p):

    x = np.array(x)
    ratio = (1 + p[1]*np.exp(-p[2]*x))/(1 + p[1]*np.exp(-p[2]*(x+1)))
    result = p[0] + np.log(ratio)
    return result


def optimum_range(xrange, effective_mass, fitting_function_name, fit_p0):

    temporal_direction_lattice_size = len(effective_mass)
    number_of_parameters = len(fit_p0)
    
    chi_square_dict = dict()
    minimum_value = 1000 # Arbitrary value
    min_key = tuple()
    fit_parameters = list()
    for upper_index_cut in range(temporal_direction_lattice_size):
        # Maintain a distance with upper index such that enough data points are used for fitting 
        for lower_index_cut in range(upper_index_cut-(number_of_parameters-1)):

            # print(lower_index_cut, upper_index_cut)

            x = np.array(xrange[lower_index_cut:upper_index_cut])
            y = effective_mass[lower_index_cut:upper_index_cut]

            fit = lsqfit.nonlinear_fit(data=(x, y), p0=fit_p0, fcn=fitting_function_name, debug=True)

            if (fit.dof == 0):
                continue

            key = (lower_index_cut, upper_index_cut)

            # print(key)

            
            # (fit.dof, fit.chi2)

            # Specify the degrees of freedom
            degrees_of_freedom = fit.dof

            # Specify the cumulative probability
            cumulative_probability = 0.95

            chi_square_value = stats.chi2.ppf(cumulative_probability, degrees_of_freedom)

            # chi_square_dict[key] = 1 - fit.Q

            p_value = 1 - fit.Q

            if (p_value < minimum_value):
                minimum_value = p_value
                min_key = key
                fit_parameters = fit

            # print(fit.chi2, chi_square_value)

    # min_value = min(chi_square_dict.values())
    # min_key = [key for key, value in chi_square_dict.items() if value == min_value][0]

    return min_key, fit_parameters

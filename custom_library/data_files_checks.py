

def filter_arrays_by_nominal_length(dictionary):
    # Get all array lengths
    array_lengths = [len(arr) for arr in dictionary.values()]

    # Take the largest length as the nominal length
    nominal_length = max(set(array_lengths))

    # Filter out arrays with different lengths
    filtered_dictionary = {key: value for key, value in dictionary.items() if len(value) == nominal_length}

    return filtered_dictionary
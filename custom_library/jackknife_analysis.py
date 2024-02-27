import numpy as np
import gvar as gv


class JackknifeAnalysis:

    def __init__(self, original_2D_array):

        self.original_2D_array = np.array(original_2D_array)

        self.sample_size = self.original_2D_array.shape[0]
        self.dataset_size = self.original_2D_array.shape[1]

        self.jackknife_average = self.jackknife_replicas()
    
    def jackknife_replicas(self):

        jackknife_replicas_2D_list = list()
        for index_to_remove in range(self.sample_size):

            reduced_original_2D_array = np.delete(self.original_2D_array, index_to_remove, axis=0)

            jackknife_replica = np.average(reduced_original_2D_array, axis=0)

            jackknife_replicas_2D_list.append(jackknife_replica)

        self.jackknife_replicas_of_original_2D_array = np.array(jackknife_replicas_2D_list)

        jackknife_average = np.average(self.jackknife_replicas_of_original_2D_array, axis=0)

        jackknife_error = np.sqrt(self.sample_size-1)*np.std(self.jackknife_replicas_of_original_2D_array, ddof=0, axis=0)

        return gv.gvar(jackknife_average, jackknife_error)
        
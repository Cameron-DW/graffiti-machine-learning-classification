# Python module to create and manipulate a DataSet for the extracted graffiti image features
# so that they can be used with the NeuralNetwork Class to train a final graffiti classifier
import numpy as np

class DataSet:
    def __init__(self, array):
        self.array = array
        self.variables = ['num_regions', 'region_colour_var',
                          'colourfulness', 'gradient_circ_var','line_amount', 'edge_amount', 'region_size_var', 'num_corners',
                          'straight_line_percent', 'avg_green', 'avg_red', 'avg_blue',
                        'class']

    def __str__(self):
        return str(self.variables)

    # Function to print an entire record of the DataSet
    def print_record(self, index):
        record_string=f"\n--RECORD [{index}]--\n"
        for variable_name, variable_value in zip(self.variables, self.array[index,:]):
            record_string+= (f"{variable_name}: {variable_value} \n")

        print(record_string)

    # Function to create an array for the input features of the DataSet
    def input(self):
        target_index = self.variables.index('class')
        input = self.array
        input_data = np.delete(input, target_index, 1)
        return input_data.astype('float32')

    # Function to create a target array from the DataSet
    def target(self):
        target = self.array
        target_index = self.get_variable_index('class')
        target_data = target[:, target_index]
        return target_data.astype('float32')

    # Function to get the number of records in the DataSet
    def num_records(self):
        return self.array.shape[0]

    # Function to get the number of variables in the DataSet
    def num_variables(self):
        return self.array.shape[1]

    # Function to get the index of a specific variable in the DataSet
    def get_variable_index(self, variable):
        return self.variables.index(variable)

    # Function to get the entire list for the values of a specific variable
    def get_variable(self, variable):
        list_variable = self.array[:, self.get_variable_index(variable)]
        return list_variable

    # Function to remove a variable from the DataSet
    def remove_variable(self, variable_list):
        data_no_vars=self.array

        for variable in variable_list:
            data_no_vars = np.delete(data_no_vars, self.variables.index(variable), 1)
            self.variables.remove(variable)
        self.array = data_no_vars

    # Function to remove an entire record from the DataSet
    def remove_record(self, index):
        self.array = np.delete(self.array, index, 0)

    # Function to get an entire record from the DataSet
    def get_record(self, index):
        return self.array[index, :]
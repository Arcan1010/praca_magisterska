import json

package_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 29,
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

training_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                    12, 13, 14, 15, 16, 17, 18, 19,
                    21, 22, 23, 24, 25, 26, 29,
                    31, 32, 33, 35, 36, 37, 38, 39,
                    41, 42, 43, 44, 45, 46, 48, 49]

# testing_indexes = [0, 34, 10, 47, 20, 30, 40]
# testing_indexes = [30]
testing_indexes = [101]

path_to_result = '..\\result\\'
path_to_fixed = '..\\new\\'
path_to_data = '..\\data\\'

def get_result_path(package_number):
    return path_to_result + str(package_number) + "_result.json"

def get_fixed_path(package_number):
    return path_to_fixed + str(package_number) + "_fixed_result.json"

def get_data_path(package_number):
    return path_to_data + "data" + str(package_number) + ".json"

def get_interior(package_number):
    return json.loads(json.load(open(get_data_path(package_number))))['interior']

def get_exterior(package_number):
    return json.loads(json.load(open(get_data_path(package_number))))['exterior']

def get_true_output(package_number):
    return json.load(open(get_result_path(package_number)))

def get_fixed_output(package_number):
    return json.load(open(get_fixed_path(package_number)))

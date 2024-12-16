import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Load in data
# Hold data in a list because its only 300 strings
data = []
with open('places.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.strip('\n'))

# print (data)

# Graphical visualization?
data_as_chart = np.zeros((300, 3))
i = 0
for pair_coords in data:
    list_of_coords = pair_coords.split(",")
    data_as_chart[i][0] = i
    data_as_chart[i][1] = list_of_coords[0]
    data_as_chart[i][2] = list_of_coords[0]
    i += 1

x_coords = data_as_chart[:, 1]
y_coords = data_as_chart[:, 2]

def dist_to_this_centroid(datapoint, target):
    """
    Given two strings, each containing two floats deliminated by a comma
    Return the distance between the two points
    """
    given_list = datapoint.split(',')
    target_list = target.split(',')

    # we are using euclidean distance to calculate distance
    distance = np.sqrt(np.power(float(given_list[0]) - float(target_list[0]), 2) + np.power(float(given_list[1]) - float(target_list[1]), 2))

    return distance

def mean_location(cluster):
    """
    Given a cluster, represented as a string containing two floats deliminated by a comma
    Return the mean location of all elements in the cluster
    """
    x_mean = 0
    y_mean = 0
    for element in cluster:
        location_as_list = element.split(',')
        x_mean += float(location_as_list[0])
        y_mean += float(location_as_list[1])
    
    x_mean = np.divide(x_mean, len(cluster))
    y_mean = np.divide(y_mean, len(cluster))

    ret_list = []
    ret_list.append(str(x_mean))
    ret_list.append(str(y_mean))

    return ','.join(ret_list)

# Decide how many clusters we want (choose k)
k = 3


# Previously we have just randomly initialized the different centroids
# first_centroid = random.choice(data)
# second_centroid = random.choice(data)
# third_centroid = random.choice(data)

# Lets try a k-means++ method for better initialization
# First centroid is initialized randomly
first_centroid = random.choice(data)
# Select next based on distance from previous centroid
max_dist = 0
candidates = []
for centroid_candidate in data:
    current_distance = dist_to_this_centroid(centroid_candidate, first_centroid)
    if current_distance > max_dist:
        max_dist = current_distance
        candidates.append(centroid_candidate)

second_centroid = candidates[-1]

# Now the first two have been initialized
# initialize the final point considering the previous two points
max_dist = 0
candidates.clear()
for centroid_candidate in data:
    current_distance_to_1 = dist_to_this_centroid(centroid_candidate, first_centroid)
    current_distance_to_2 = dist_to_this_centroid(centroid_candidate, second_centroid)
    current_distance = current_distance_to_1 + current_distance_to_2
    if current_distance > max_dist:
        max_dist = current_distance
        candidates.append(centroid_candidate)

third_centroid = candidates[-1]


new_first_centroid = 0
new_second_centroid = 0
new_third_centroid = 0
while (first_centroid != new_first_centroid and second_centroid != new_second_centroid and third_centroid != new_third_centroid):
    if new_first_centroid != 0:
        first_centroid = new_first_centroid
        second_centroid = new_second_centroid
        third_centroid = new_third_centroid

    # Create a way to remember the distance between all observations and each of the k centroids
    # This array will contain a 300 X 3 matrix where the columns will be: "dist_to_first_centroid" "dist_to_second_centroid" "dist_to_third_centroid"
    dist_array = np.zeros((300, 3))
    # Repeat the following until the centroids do not change position
    # Calculate the distance of all observation to each of the k centroids
    index = 0
    for datapoint in data:
        dist_array[index][0] = dist_to_this_centroid(datapoint, first_centroid)
        dist_array[index][1] = dist_to_this_centroid(datapoint, second_centroid)
        dist_array[index][2] = dist_to_this_centroid(datapoint, third_centroid)
        index += 1


    # Assign observations to nearest centroid
    first_cluster = []
    second_cluster = []
    third_cluster = []
    dist_index = 0
    for distances in dist_array:
        cluster_id = np.argmin(distances)
        if cluster_id == 0:
            first_cluster.append(data[dist_index])
        if cluster_id == 1:
            second_cluster.append(data[dist_index])
        if cluster_id == 2:
            third_cluster.append(data[dist_index])

        # This code only exists so that debugging could be done
        # if dist_index == 6:
        #     break

        dist_index += 1

    new_first_centroid = mean_location(first_cluster)
    new_second_centroid = mean_location(second_cluster)
    new_third_centroid = mean_location(third_cluster)

# print(first_cluster)
# print(second_cluster)
# print(third_cluster)

file_as_list = []
output_index = 0
for ele in data:
    if ele in first_cluster:
        file_as_list.append(str(output_index) + " 0" + "\n")
    if ele in second_cluster:
        file_as_list.append(str(output_index) + " 1" + "\n")
    if ele in third_cluster:
        file_as_list.append(str(output_index) + " 2" + "\n")    
    output_index += 1


with open('clusters.txt', 'w') as f:
    f.writelines(file_as_list)
    print('Output file created: clusters.txt')

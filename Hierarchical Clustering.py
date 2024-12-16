# Submit this file to Gradescope
from typing import List
import math
from collections import defaultdict
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.

class Solution:
    def dist_between_points(self, p1, p2):
        """
        Given two points (represented as a list containing two floats)
        Return the distance between them (represented as a float)
        """
        return math.sqrt(((p1[0] - p2[0])**2) +(p1[1] - p2[1])**2)
    
    def find_element(self, matrix, element):
        for i, row in enumerate(matrix):
            if element in row:
                return (i, row.index(element))
        return None
    
    def num_unique_elements(self, list_in):
        """
        Given a list of elements, determine how many unique elements are in it
        """
        temp_set = set(list_in)
        return len(temp_set)
    
    def reassign_values(self, list_in):
        unique_values = sorted(set(list_in))
        value_to_new_value = {value:index for index, value in enumerate(unique_values)}
        reassigned_list = [value_to_new_value[value] for value in list_in]

        return reassigned_list

    def group_floats_by_indices(self, index_list, float_list):
        # Initialize a dictionary to hold lists of floats grouped by index
        grouped_floats = defaultdict(list)
    
        # Iterate through index_list and float_list simultaneously
        for index, floats in zip(index_list, float_list):
            # Add the float pair to the group corresponding to the index
            grouped_floats[index].append(floats)
    
        # Convert defaultdict to a regular list of lists, sorted by the index
        result = [grouped_floats[i] for i in sorted(grouped_floats.keys())]
    
        return result


    # def dist_between_clusters_single(self, cluster1, cluster2):
    #     """
    #     Calculates the single-linkage distance between clusters
    #     """
    #     min_dist = 0
    #     for p1 in cluster1:
    #         for p2 in cluster2:
    #             curr_dist = self.dist_between_points(p1, p2)
    #             if curr_dist < min_dist:
    #                 min_dist = curr_dist
    #     return min_dist

    def dist_between_clusters_complete(self, cluster1, cluster2):
        """
        Calculates the complete-linkage distance between clusters
        """
        max_dist = 0
        for p1 in cluster1:
            for p2 in cluster2:
                curr_max_dist = self.dist_between_points(p1, p2)
                if curr_max_dist > max_dist:
                    max_dist = curr_max_dist
        return max_dist
    
    def update_dist_matrix_complete(self, input_data, clusters):
        """
        Recalculate the distance matrix, given the input data and given class labels representing which points are clustered together
        """
        dist_matrix = []
        for i, clus1 in enumerate(clusters):
            dist_matrix.append([])
            for j, clus2 in enumerate(clusters):
                first_cluster = []
                second_cluster = []
                for ele in clus1:
                    first_cluster.append(input_data[ele])
                for elem in clus2:
                    second_cluster.append(input_data[elem])
                dist_matrix[i].insert(j, self.dist_between_clusters_complete(first_cluster, second_cluster))

        return dist_matrix
    
    def dist_between_clusters_average(self, cluster1, cluster2):
        """
        Calculates the average-linkage distance between clusters
        """
        # Is this the correct way to calculate average distance between clusters?
        total_dist = 0
        for p1 in cluster1:
            for p2 in cluster2:
                total_dist += self.dist_between_points(p1, p2)
        
        avg_dist = total_dist / (len(cluster1) * len(cluster2))
        return avg_dist
    
    def update_dist_matrix_average(self, input_data, clusters):
        """
        Recalculate the distance matrix, given the input data and given class labels representing which points are clustered together
        """
        dist_matrix = []
        for i, clus1 in enumerate(clusters):
            dist_matrix.append([])
            for j, clus2 in enumerate(clusters):
                first_cluster = []
                second_cluster = []
                for ele in clus1:
                    first_cluster.append(input_data[ele])
                for elem in clus2:
                    second_cluster.append(input_data[elem])
                dist_matrix[i].insert(j, self.dist_between_clusters_average(first_cluster, second_cluster))

        return dist_matrix

    def find(self, parent, i):
    # Recursively find the root of i with path compression
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent, x, y):
        # Unite two elements by linking their roots
        root_x = self.find(parent, x)
        root_y = self.find(parent, y)
        if root_x != root_y:
            parent[root_y] = root_x

    def unify_connected_components(self, lst):
        n = len(lst)
    
        # Initialize each element as its own parent
        parent = list(range(n))
    
        # Perform union operations for all connections
        for i in range(n):
            if lst[i] != i:  # Only unify if they are not pointing to themselves
                self.union(parent, i, lst[i])
    
        # Make sure all elements have the same root in their component
        for i in range(n):
            parent[i] = self.find(parent, i)
    
        return parent

    def dist_between_clusters_single(self, cluster1, cluster2):
        """
        Calculates the single-linkage distance between clusters
        """
        min_dist = float('inf')
        for p1 in cluster1:
            for p2 in cluster2:
                curr_dist = self.dist_between_points(p1, p2)
                if curr_dist < min_dist:
                    min_dist = curr_dist
        return min_dist

    def update_dist_matrix_single(self, clusters: List[List[int]], input_data):
        """
        Given a list of lists of ints, return the single-link distance matrix
        """
        dist_matrix = []
        for row_id, clus1 in enumerate(clusters):
            dist_matrix.append([])
            for col_id, clus2 in enumerate(clusters):
                first_cluster = []
                second_cluster = []
                for ele in clus1:
                    first_cluster.append(input_data[ele])
                for elem in clus2:
                    second_cluster.append(input_data[elem])
                dist_matrix[row_id].insert(col_id, self.dist_between_clusters_single(first_cluster, second_cluster))

        return dist_matrix

            
    def reassign_cluster_id(self, clusters, length):
        """
        Given a list of lists containing ints, return a 1-D array where the value at the index is turned into that index
        """
        # So I get the right clusters, but this doesn't do what I think it should
        return_list = [None] * length
        for i, lst in enumerate(clusters):
            for element in lst:
                if 0 <= element < length:
                    return_list[element] = i

        return return_list



    def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
        """Single link hierarchical clustering
        Args:
        - X: 2D input data
        - K: the number of output clusters
        Returns:
        A list of integers (range from 0 to K - 1) that represent class labels.
        The number does not matter as long as the clusters are correct.
        For example: [0, 0, 1] is treated the same as [1, 1, 0]"""
        
        # Start by assigning each of the inputs a different class label (i.e., put each point in a different cluster)
        class_labels = [i for i in range(len(X))]
        clusters = [[i] for i in range(len(X))]
        # print(class_labels)

        # Create distance matrix for when each point is its own cluster
        dist_matrix = []
        for i in range(len(X)):
            dist_matrix.append([])
            for j in range(len(X)):
                dist_matrix[i].insert(j, self.dist_between_points(X[i], X[j]))
        # print(dist_matrix)


        # Make sure this can handle it when there are distances that are the same

        # Make this a set
        remembered_min_distances = set()
        # Loop until all elements in class_labels are in the range of 0 : k-1
        while (len(clusters)) != K:
            cluster_id = []
            # # Instead of iterating over the distance matrix, iterate over the clusters
            # for c1_id, clus1 in enumerate(clusters):
            #     for c2_id, clus2 in enumerate(clusters):
            #         # Determine the smallest distance between this cluster and all others
            #         min_dist = float('inf')
                    
            #         curr_dist = self.dist_between_clusters_single()

            #         if curr_dist < min_dist:
            #             min_dist = curr_dist
            #             cluster_id.append([c1_id, c2_id])
            
            # indicies_to_merge = cluster_id[-1]
            # clusters[indicies_to_merge[0]].extend(clusters[indicies_to_merge[1]])
            # del clusters[indicies_to_merge[1]]

                

            min_dist = float('inf')
            for row_id, row in enumerate(dist_matrix):
                for col_id, dist in enumerate(row):
                    if dist < min_dist and dist != 0 and dist not in remembered_min_distances:
                        min_dist = dist
                        # Remeber index in some way
                        cluster_id.append([row_id, col_id])
                       

            if min_dist not in remembered_min_distances:
                remembered_min_distances.add(min_dist)
            
            # # indicies_to_merge = self.find_element(dist_matrix, min_dist)

            # Mergining operation
            indicies_to_merge = cluster_id[-1]
            clusters[indicies_to_merge[0]].extend(clusters[indicies_to_merge[1]])
            del clusters[indicies_to_merge[1]]
            # print(len(clusters))

            # Update distance matrix with new clusters
            dist_matrix = self.update_dist_matrix_single(clusters, X)

            # class_labels[indicies_to_merge[1]] = class_labels[indicies_to_merge[0]]
            # print(class_labels)

        # Reassign class labels such that they fall into the range of 0 to k - 1
        # return self.reassign_values(class_labels)
        # print(clusters)
        return self.reassign_cluster_id(clusters, len(X))
    

    def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
        """Complete link hierarchical clustering"""
        # Start by assigning each of the inputs a different class label (i.e., put each point in a different cluster)
        class_labels = [i for i in range(len(X))]
        clusters = [[i] for i in range(len(X))]
        # print(class_labels)

        # Create distance matrix
        dist_matrix = []
        for i in range(len(X)):
            dist_matrix.append([])
            for j in range(len(X)):
                dist_matrix[i].insert(j, self.dist_between_points(X[i], X[j]))


        # remembered_min_distances = []
        remembered_min_distances = set()
        # Loop until all elements in class_labels are in the range of 0 : k-1
        while (len(clusters)) != K:
            # We want to add the most similar clusters together
            # Use the diameter of the two clusters as the similarity measure
            min_dist = float('inf')
            cluster_id = []
            for ri, row in enumerate(dist_matrix):
                for rj, dist in enumerate(row):
                    if dist < min_dist and dist != 0 and dist not in remembered_min_distances:
                        min_dist = dist
                        cluster_id.append([ri, rj])

            # if min_dist not in remembered_min_distances:
            #     remembered_min_distances.append(min_dist)
            
            # There is an issue with how we combine clusters, sometimes they are clustered together but don't have the same root

            if min_dist not in remembered_min_distances:
                remembered_min_distances.add(min_dist)
            # Merge clusters
            indicies_to_merge = cluster_id[-1]
            clusters[indicies_to_merge[0]].extend(clusters[indicies_to_merge[1]])
            del clusters[indicies_to_merge[1]]
            # print(class_labels)

            # Update the distance matrix
            dist_matrix = self.update_dist_matrix_complete(input_data=X, clusters=clusters)
            # print(dist_matrix)
            # Consolidate clusters
            # temp = self.unify_connected_components(class_labels)
            # print(temp)
            # class_labels = temp


        # Reassign class labels such that they fall into the range of 0 to k - 1
        return self.reassign_cluster_id(clusters, len(X))

    def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
        """Average link hierarchical clustering"""
        # Start by assigning each of the inputs a different class label (i.e., put each point in a different cluster)
        class_labels = [i for i in range(len(X))]
        clusters = [[i] for i in range(len(X))]
        # print(class_labels)

        # Create distance matrix
        dist_matrix = []
        for i in range(len(X)):
            dist_matrix.append([])
            for j in range(len(X)):
                dist_matrix[i].insert(j, self.dist_between_points(X[i], X[j]))

        remembered_min_distances = set()
        cluster_id = []
        # Loop until all elements in class_labels are in the range of 0 : k-1
        while (len(clusters)) != K:
            # Determine smallest distance in distance matrix
            min_dist = float('inf')
            for rid, row in enumerate(dist_matrix):
                for cid, dist in enumerate(row):
                    if dist < min_dist and dist != 0 and dist not in remembered_min_distances and rid != cid:
                        min_dist = dist
                        cluster_id.append([rid, cid])

            if min_dist not in remembered_min_distances:
                remembered_min_distances.add(min_dist)

            # Merging Operation
            indicies_to_merge = cluster_id[-1]
            clusters[indicies_to_merge[0]].extend(clusters[indicies_to_merge[1]])
            del clusters[indicies_to_merge[1]]
            # print(class_labels)
            # Update the distance matrix
            dist_matrix = self.update_dist_matrix_average(input_data=X, clusters=clusters)
            # print(class_labels)

        # Reassign class labels such that they fall into the range of 0 to k - 1
        return self.reassign_cluster_id(clusters, len(X))
        
  
# input1 = [[7.8122, 7.0391], [-90.3764, 14.7628], [152.8991, -30.4529], [8.2569, 49.8364], [145.4259, -37.8743]]
# input2 = [[-69.1381, 11.3665], [-92.7664, 16.7984], [37.1320, -1.7395], [127.0042, 36.7836], [113.9475, 23.0878], [-93.6641, 15.9575], [-89.5223, 32.0335], [10.8291, 45.4762], [144.3030, -36.7713],
# [29.2284, -1.6741], [122.3640, 36.9858], [19.9728, 49.9462], [-0.8175, 46.5392], [9.6104, 7.5067], [-35.2214, -6.4392], [-14.0333, 10.1667],
# [117.6688, 29.6952], [151.2368, -33.8795], [27.8920, -32.5770], [-0.7055, 45.9320]]
# Current implementation has elements at index 10 and 12 in the wrong clusters


# input3 = [[115.9098, -32.0000],
# [-2.8812, 43.3054],
# [5.0167, 51.9825],
# [153.3167, -28.8167],
# [21.1569, 45.9109],
# [33.1550, 1.1517],
# [109.1333, 43.1500],
# [115.0926, -8.1256],
# [-79.0873, 9.1843],
# [15.5606, -12.8525]]

# test = Solution()
# # print(Solution.hclus_single_link(self=test, X=input1, K = 2))
# print(Solution.hclus_complete_link(self=test, X=input2, K = 2))
# print(Solution.hclus_average_link(self=test, X=input3, K = 2))
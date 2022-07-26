# Now that we've covered regression and classification
# it's time to talk about clustering data!

# Clustering is a Machine Learning technique that
# involves the grouping of data points.
# In theory, data points that are in the same group
# should have similar properties and/or features,
# while data points in different groups should have
# highly dissimilar properties and/or features.
# (https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

# Unfortunately there are issues with the current version of TensorFlow
# and the implementation for KMeans.
# This means we cannot use KMeans without writing
# the algorithm from scratch. We aren't quite at that level yet,
# so we'll just explain the basics of clustering for now.

# Basic Algorithm for K-Means.
# Step 1: Randomly pick K points to place K centroids
# Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
# Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
# Step 4: Reassign every point once again to the closest centroid.
# Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.
# kmeans
This is a kmeans implementation from scratch using Numpy on the Iris dataset.

The script provides the following optional arguments:
- Choose the initialisation type for kmeans - [*kmeans++*](https://en.wikipedia.org/wiki/K-means%2B%2B) or random([*forgy*](https://en.wikipedia.org/wiki/K-means_clustering#Algorithms) (see initialisation methods)) initialisation. Default is random(forgy) initialisation.
- Choose the distance metric to use. Default is the  `L2` norm.
- Choose the number of clusters to perform kmeans or select the optimal number of clusters using the silhouette score.

To run the script, make sure you have the latest version of Numpy installed. The script uses the iris dataset, which is present in the data folder.

The following example will perform kmeans on the Iris dataset using the forgy initialisation using 3 clusters and the L2 norm:

> `python cluster.py -i forgy -p 2 -k 3`

The following example will perform kmeans on the Iris dataset using the forgy initialisation and display the silhouette score graph for different number of clusters, using the L2 norm:

> `python cluster.py -i forgy -p 2 -k select`

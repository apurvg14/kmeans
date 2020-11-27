# kmeans clustering in Numpy over the iris dataset
This is a kmeans implementation from scratch using Numpy on the Iris dataset.

The script provides the following optional arguments, which can also be seen by doing `python cluster.py -h`:
- Choose the distance metric to use. Default is the  `L2` norm.
- Choose the number of clusters to perform kmeans.
- Choose the cluster initialisation type for kmeans - [*kmeans++*](https://en.wikipedia.org/wiki/K-means%2B%2B) or random ([*forgy*](https://en.wikipedia.org/wiki/K-means_clustering#Algorithms)) initialisation. Default is random(forgy) initialisation.
- Perform Silhouette analysis to get the optimal number of clusters for kmeans
- Specify the kind of data transform you want to apply.

To run the script, make sure you have the latest version of Numpy installed. The script uses the iris dataset, which is present in the `data` folder. All plots get saved in the `outputs` folder.

To generate the results/plots, do: `python runner.py`, and observe the terminal output as well as the images in the `outputs` folder

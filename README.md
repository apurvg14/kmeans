# kmeans clustering in Numpy over the iris dataset
This is a kmeans implementation from scratch using Numpy on the Iris dataset.

The script `cluster.py` provides the following optional arguments, which can also be seen by doing `python cluster.py -h`:
- Choose the distance metric to use. Default is the  `L2` norm.
- Choose the number of clusters to perform kmeans.
- Choose the cluster initialisation type for kmeans - [*kmeans++*](https://en.wikipedia.org/wiki/K-means%2B%2B) or random ([*forgy*](https://en.wikipedia.org/wiki/K-means_clustering#Algorithms)) initialisation. Default is random(forgy) initialisation.
- Perform Silhouette analysis to get the optimal number of clusters for kmeans
- Specify the kind of data transform you want to apply.

To run the script, make sure you have the latest version of Numpy installed. The script uses the iris dataset, which is present in the `data` folder. 

To generate the results/plots, do: `python runner.py`, and observe the terminal output (refer to `log.txt`). All images of the points plotted in multivariate space and color coded based on classification, get saved in the `plots` folder. All prediction csvs, which are the source dataset with appended columns of classifications, get stored in the `predictions` folder. Both these folders contain results for both `L1` and `L2` norms. 

Based on a [Silhouette analysis](https://en.wikipedia.org/wiki/Silhouette_(clustering)) of the dataset, it was determined that `k=5` clusters seem optimal, as the silhouette elbow is achieved at this point. Thus, I have also included results for `k=5`. Please refer to the Silhouette score curves in the `plots` folder for the same.

Note that `runner.py` is just a wrapper around `cluster.py` to control the various hyperparamters. Feel free to tweak and play around with these numbers.

The accuracy metrics obtained with 3 clusters are as follows:

Norm | Kmeans++, Raw data | Forgy, Raw data | Kmeans++, Standardised data | Forgy, Standardised data |
--- | --- | --- | --- | --- |
L1 norm | 95% | 55% | 90% | 60% |
L2 norm | 95% | 55% | 85% | 65% |

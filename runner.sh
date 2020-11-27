# Kmeans++ initialisation, L2 norm, 3 clusters
python cluster.py -p 2 -k 3 -m unchanged -i plusplus -s
python cluster.py -p 2 -k 3 -m normalised -i plusplus -s
python cluster.py -p 2 -k 3 -m standardised -i plusplus -s

# Forgy initialisation, L2 norm, 3 clusters
python cluster.py -p 2 -k 3 -m unchanged -i forgy -s
python cluster.py -p 2 -k 3 -m normalised -i forgy -s
python cluster.py -p 2 -k 3 -m standardised -i forgy -s

###################################################################

# Kmeans++ initialisation, L1 norm, 3 clusters
python cluster.py -p 1 -k 3 -m unchanged -i plusplus -s
python cluster.py -p 1 -k 3 -m normalised -i plusplus -s
python cluster.py -p 1 -k 3 -m standardised -i plusplus -s

# Forgy initialisation, L1 norm, 3 clusters
python cluster.py -p 1 -k 3 -m unchanged -i forgy -s
python cluster.py -p 1 -k 3 -m normalised -i forgy -s
python cluster.py -p 1 -k 3 -m standardised -i forgy -s

###################################################################

# Runs using num clusters = 5 as obtained from the silhouette curve

# Kmeans++ initialisation, L2 norm
python cluster.py -p 2 -k 5 -m unchanged -i plusplus -s
python cluster.py -p 2 -k 5 -m normalised -i plusplus -s
python cluster.py -p 2 -k 5 -m standardised -i plusplus -s

# Forgy initialisation, L2 norm
python cluster.py -p 2 -k 5 -m unchanged -i forgy -s
python cluster.py -p 2 -k 5 -m normalised -i forgy -s
python cluster.py -p 2 -k 5 -m standardised -i forgy -s

# Kmeans++ initialisation, L1 norm
python cluster.py -p 1 -k 5 -m unchanged -i plusplus -s
python cluster.py -p 1 -k 5 -m normalised -i plusplus -s
python cluster.py -p 1 -k 5 -m standardised -i plusplus -s

# Forgy initialisation, L1 norm
python cluster.py -p 1 -k 5 -m unchanged -i forgy -s
python cluster.py -p 1 -k 5 -m normalised -i forgy -s
python cluster.py -p 1 -k 5 -m standardised -i forgy -s

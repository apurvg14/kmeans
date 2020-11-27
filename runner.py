import os
import time

start = time.time()

# Kmeans++ initialisation, L2 norm, 3 clusters
os.system('python -W ignore cluster.py -p 2 -k 3 -m unchanged -i plusplus -s')
# os.system('python -W ignore cluster.py -p 2 -k 3 -m normalised -i plusplus -s')
os.system('python -W ignore cluster.py -p 2 -k 3 -m standardised -i plusplus -s')

# Forgy initialisation, L2 norm, 3 clusters
os.system('python -W ignore cluster.py -p 2 -k 3 -m unchanged -i forgy -s')
# os.system('python -W ignore cluster.py -p 2 -k 3 -m normalised -i forgy -s')
os.system('python -W ignore cluster.py -p 2 -k 3 -m standardised -i forgy -s')

###################################################################

# Kmeans++ initialisation, L1 norm, 3 clusters
os.system('python -W ignore cluster.py -p 1 -k 3 -m unchanged -i plusplus -s')
# os.system('python -W ignore cluster.py -p 1 -k 3 -m normalised -i plusplus -s')
os.system('python -W ignore cluster.py -p 1 -k 3 -m standardised -i plusplus -s')

# Forgy initialisation, L1 norm, 3 clusters
os.system('python -W ignore cluster.py -p 1 -k 3 -m unchanged -i forgy -s')
# os.system('python -W ignore cluster.py -p 1 -k 3 -m normalised -i forgy -s')
os.system('python -W ignore cluster.py -p 1 -k 3 -m standardised -i forgy -s')

###################################################################

# Runs using num clusters = 5 as obtained from the silhouette curve

# Kmeans++ initialisation, L2 norm
os.system('python -W ignore cluster.py -p 2 -k 5 -m unchanged -i plusplus -s')
# os.system('python -W ignore cluster.py -p 2 -k 5 -m normalised -i plusplus -s')
os.system('python -W ignore cluster.py -p 2 -k 5 -m standardised -i plusplus -s')

# Forgy initialisation, L2 norm
os.system('python -W ignore cluster.py -p 2 -k 5 -m unchanged -i forgy -s')
# os.system('python -W ignore cluster.py -p 2 -k 5 -m normalised -i forgy -s')
os.system('python -W ignore cluster.py -p 2 -k 5 -m standardised -i forgy -s')

# Kmeans++ initialisation, L1 norm
os.system('python -W ignore cluster.py -p 1 -k 5 -m unchanged -i plusplus -s')
# os.system('python -W ignore cluster.py -p 1 -k 5 -m normalised -i plusplus -s')
os.system('python -W ignore cluster.py -p 1 -k 5 -m standardised -i plusplus -s')

# Forgy initialisation, L1 norm
os.system('python -W ignore cluster.py -p 1 -k 5 -m unchanged -i forgy -s')
# os.system('python -W ignore cluster.py -p 1 -k 5 -m normalised -i forgy -s')
os.system('python -W ignore cluster.py -p 1 -k 5 -m standardised -i forgy -s')


print("Time taken:", round(time.time() - start, 2), "seconds.")
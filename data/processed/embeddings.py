import numpy as np

embeddings = np.random.uniform(-1, 1, (6, 16)) 

np.save("embeddings.npy", embeddings)

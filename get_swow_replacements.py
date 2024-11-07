import numpy as np
import pandas as pd
import csv

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_folder = lbl.llms_activations
make_dir(output_folder)

model_name = 'swow'

# code from https://spotintelligence.com/2023/11/27/glove-embedding/
# Load SWOW embeddings into a dictionary
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


swow_embeddings = load_embeddings(lbl.swow_embeddings_path)


glove_embeddings = load_embeddings(lbl.glove_embeddings_path)
glove_keys = list(glove_embeddings.keys())



swow_missing = []
# filter SWOW missing words
with open(lbl.swow_missing_path , 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        swow_missing.append(values[0])




# normalized glove embeddings
norm_glove_embeddings = np.vstack(list(glove_embeddings.values()))
norm_glove_embeddings = norm_glove_embeddings / np.linalg.norm(norm_glove_embeddings, axis=1, keepdims=True)

# filter GloVE by SWOW's vocabulary
idx = [key in swow_embeddings.keys() for key in glove_keys]
norm_glove_embeddings_filtered = norm_glove_embeddings[idx,:]
glove_keys_filtered = [val for id,val in enumerate(glove_keys) if idx[id] ]
print(glove_keys_filtered)

pairs = {}
for missing in swow_missing:
    if missing in glove_keys:
        cosine_sims = np.dot(norm_glove_embeddings[glove_keys.index(missing),:],norm_glove_embeddings_filtered.T)
        pairs[missing] = glove_keys_filtered[ np.argmax(cosine_sims) ]


with open("swow_replacements.csv","w") as file:
    writer = csv.writer(file)

    for key, value in pairs.items():
        writer.writerow([key,value])

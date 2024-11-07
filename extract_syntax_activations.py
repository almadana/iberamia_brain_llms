import numpy as np
import pandas as pd
import os
import joblib
from tqdm import tqdm
import nltk
from nltk.corpus import cmudict

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_folder = lbl.llms_activations
make_dir(output_folder)

model_name = 'syntax'


filename = os.path.join(lbl.annotation_folder, 'lppEN_word_information.csv')
df_word_onsets = pd.read_csv(filename)

df_word_onsets = df_word_onsets.drop([3919,6775,6781]) 
# 3919: adhoc removal of repeated line with typo
# 6775: mismatch with full text

word_list_runs = []
onsets_offsets_runs = []
features_runs = []

word_list_runs = []
onsets_offsets_runs = []


pos_labels = df_word_onsets.pos.unique()
label_to_index = {label: idx for idx, label in enumerate(pos_labels)}
df_word_onsets['label_index'] = df_word_onsets['pos'].map(label_to_index)
print(label_to_index)

#read the phon feature matrix
filename = os.path.join('phonemes_features_matrix.csv')
df_phone = pd.read_csv(filename)
df_phone.set_index('phon',inplace=True)
phone_dim = df_phone.shape[1]
#phone_dict = df_phone.to_dict('index')

cmu_dict = cmudict.dict()

def get_phonemes(word):
    word = word.lower()
    if word in cmu_dict:
        # Get the first pronunciation
        phonemes = cmu_dict[word][0]
        # Remove stress markers
        phonemes_no_stress = [''.join([char for char in phoneme if not char.isdigit()]) for phoneme in phonemes]
        return phonemes_no_stress
    else:
        print(f"No pronunciation found for '{word}'.")
        return None

def check_phon(word):
    if word=="hasnt":
        return("hasn't")
    elif word=="na\ive":
        return("naïve")
    return(word)

for run in range(lbl.n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    logfreq_tmp = df_word_onsets_run.logfreq.to_numpy()


    # One-hot encode using NumPy
    one_hot = np.zeros((df_word_onsets.shape[0] , len(pos_labels)), dtype=int)
    one_hot[np.arange(df_word_onsets_run.shape[0]), df_word_onsets_run['label_index']] = 1
    pos_tmp = one_hot
    top_tmp = df_word_onsets_run.top_down.to_numpy()
    bottom_tmp = df_word_onsets_run.bottom_up.to_numpy()
    left_tmp = df_word_onsets_run.left_corner.to_numpy()


    word_list = []
    onsets = []
    offsets = []
    features = []


    for idx_word, (word, onset, offset,logfreq,pos,top,bottom,left) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp,logfreq_tmp,pos_tmp,top_tmp,bottom_tmp,left_tmp)):
        if isinstance(word, str):
            word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)

            # phone encoding
            phonemes = get_phonemes(check_phon(word))
            if phonemes:
                phone_features = np.sum(df_phone.loc[phonemes].to_numpy(),axis=0) # sum of phonetic features of a word's phonemes
            else:
                phone_features = np.zeros(phone_dim)
                #print(phone_features)
            #print(phone_features)
            #feats = [logfreq] + [top]  + [bottom] + [left] + pos #+ phone_features.tolist()
            #if not len(features)==35:
             #   print("Che!")
            vector  = np.concatenate((pos,phone_features))
            features.append( [logfreq]  + vector + [top]  + [bottom] + [left] )

            # logfreqs.append(logfreq)
            # poss.append(pos)
            # tops.append(top)
            # bottoms.append(bottom)
            # lefts.append(left)

    #print(features)
    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)
    features_runs.append( [features])
# n_runs x 1 x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(model_name))
with open(filename, 'wb') as f:
     joblib.dump(features_runs, f, compress=4)

if not os.path.exists(os.path.join(lbl.llms_activations, 'onsets_offsets.gz')):
    filename = os.path.join(output_folder, 'onsets_offsets.gz')
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=4)

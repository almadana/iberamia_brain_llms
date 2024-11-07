import nltk
from nltk.corpus import cmudict
import panphon
import numpy as np
import pandas as pd

# Step 1: Obtain the Full List of Phonemes
nltk.download('cmudict')
cmu_dict = cmudict.dict()

all_phonemes = set()
for word in cmu_dict:
    for pronunciation in cmu_dict[word]:
        for phoneme in pronunciation:
            phoneme = ''.join([char for char in phoneme if not char.isdigit()])
            all_phonemes.add(phoneme)

all_phonemes = sorted(all_phonemes)

print(f"Total unique phonemes: {len(all_phonemes)}")
print("List of phonemes:")
print(all_phonemes)

# Step 2: Map ARPABET Phonemes to IPA Symbols
arpabet_to_ipa = {
    # Vowels
    'AA': 'ɑ',
    'AE': 'æ',
    'AH': 'ʌ',  # Also 'ə' in unstressed positions
    'AO': 'ɔ',
    'AW': 'aʊ',
    'AX': 'ə',
    'AY': 'aɪ',
    'EH': 'ɛ',
    'ER': 'ɛ',  # Stressed # it should be ɚ or ɝ
    'EY': 'eɪ',
    'IH': 'ɪ',
    'IY': 'i',
    'OW': 'oʊ',
    'OY': 'ɔɪ',
    'UH': 'ʊ',
    'UW': 'u',
    'UX': 'ʉ',  # Rarely used
    # Consonants
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'EL': 'l̩',  # Syllabic L
    'EM': 'm̩',  # Syllabic M
    'EN': 'n̩',  # Syllabic N
    'F': 'f',
    'G': 'ɡ',
    'HH': 'h',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'P': 'p',
    'Q': 'ʔ',  # Glottal stop
    'R': 'ɹ',   # American English R
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'V': 'v',
    'W': 'w',
    'WH': 'ʍ',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ',
}


#relevant features
feature_set = ['syl','son','cont','ant','lat','nas','strid','round','ant','cor','distr','lab','hi','lo','back','voi','tense']

# Step 3: Extract Phonological Features Using Panphon
ft = panphon.FeatureTable()


phoneme_features_list = []
phoneme_labels = []

for arpabet_phoneme in all_phonemes:
    ipa_symbol = arpabet_to_ipa.get(arpabet_phoneme)
    if not ipa_symbol:
        print(f"No IPA mapping for phoneme '{arpabet_phoneme}'. Skipping.")
        continue

    # Handle diphthongs and affricates by splitting them
    ipa_symbols = ft.segs_safe(ipa_symbol)
    print(ipa_symbols)
    # Initialize a list to store vectors for each IPA symbol
    vectors = ''.join(ipa_symbols)
    features = ft.word_array(feature_set,vectors)
    if vectors:
        # Average the vectors if multiple symbols
        aggregated_vector = np.mean(features, axis=0)
        print(aggregated_vector.shape)
        phoneme_features_list.append(aggregated_vector)
        phoneme_labels.append(arpabet_phoneme)
    else:
        print(f"No features found for IPA symbol(s) in '{ipa_symbol}'. Skipping.")





# Convert the list to a NumPy array
feature_matrix = np.array(phoneme_features_list)

# Create a DataFrame for better visualization
df = pd.DataFrame(feature_matrix, index=phoneme_labels, columns=feature_set)

print(df.head())

# Optionally, save the DataFrame to a CSV file
df.to_csv('phonemes_features_matrix.csv')

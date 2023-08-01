from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def label_normalise(text, stop_words):
    '''
    @param: text: str, the text needed to be tokenized
    '''
    text = text.replace('_', ' ') # replace '_' with white space
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # CamelCase tokenization
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, stop_words))) # remove stopwords
    text = re.sub(pattern, '', text.lower())
    # remove punctuations
    filtered_text = re.sub(r'[^\w\s]', '', text)
    
    return ''.join(filtered_text.lower().split())


def string_matching(source_labels, target_labels):
    '''
    setting confidence_score of those exact matching labels to 1 by default.
    It is equivalent to do exact name filtering first, then do embedding module to get final confidence for each correspondence.
    @param: 
            source_labels: List[str], list of labels of source ontology classes
            target_labels: List[str], list of labels of target ontology classes
    '''
    stop_words = stopwords.words('english')
    simi_matrix = np.zeros((len(source_labels), len(target_labels)))
    for i,s in enumerate(tqdm(source_labels, desc='string matching')):
        for j,t in enumerate(target_labels):
            # pre_process
            src = label_normalise(s, stop_words)
            tar = label_normalise(t, stop_words)
            # print(s,t)
            if tar == src :
                simi_matrix[i, j] = 1
                break
    return csr_matrix(simi_matrix)
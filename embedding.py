import wordsegment
wordsegment.load()
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm


class Embedding:
    def __init__(self, model, name) -> None:
        self.model = model
        self.model_name = name

    def _tokenize(self, text):
        '''
        @param: text: the text needed to be tokenized
                model: the pre-trained model used to get word embedding
        '''
        text = text.replace('-', ' ') # replace '-' with white space
        # tokenization
        tokens = text.split() # white space split
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords
        tokens_filtered = []
        for token in tokens:
            token = token.strip(string.punctuation)
            if token in stop_words:
                continue
            elif not re.match(r'[A-Z][a-z]+', token) and token.lower() in self.model:
                tokens_filtered.append(token.lower())
                continue

            # remove punctuations
            split_tokens = re.split(r'[^\w\s]', token)
            if len(split_tokens) > 1:
                for w in split_tokens:
                    if w in stop_words:
                        continue
                    elif len(w) > 0  and w.lower() in self.model:
                        tokens_filtered.append(w.lower())
                continue

            # some special situations
            split_words = re.split(r'(?<=[A-Z])(?=[A-Z][a-z][a-z]+)', token) # CDFormat -> CD Format; RDFs -> RDFs
            if len(split_words) > 1:
                for w in split_words:
                    ws = re.findall(r'[A-Z][a-z]+', w) # to deal with DJMixAlbum -> DJ Mix Album
                    if len(ws) > 1:
                        split_words += ws
                    if w in stop_words:
                        continue
                    elif len(w) > 0  and w.lower() in self.model:
                        tokens_filtered.append(w.lower())
                continue

            split_words = re.findall(r'[A-Z][a-z]+', token) # SingleAlbum -> Single Album
            match = re.match(r'^[A-Z]+[a-z]$', token) # to avoid RDFs being splited
            if split_words and not match:
                for w in split_words:
                    if w.lower() in stop_words:
                        continue
                    elif len(w) > 0 and w.lower() in self.model:
                        tokens_filtered.append(w.lower())
                continue

            # split consecutive words: 'creativeworkseries' or 'dancegroup'
            segments = wordsegment.segment(token)
            if len(segments) > 1:
                for seg in segments:
                    if seg in stop_words:
                        continue
                    elif seg in self.model:
                        tokens_filtered.append(seg)
                        continue
        
        return tokens_filtered
    
    
    def _w2v_embedding(self, text, pooling='mean'):

        tokens = self._tokenize(text)
        dim = self.model.vector_size

        embeddings = []
        if tokens:
            for token in tokens:
                embeddings.append(self.model[token])
        else: # No useful tokens in text
            embeddings.append(np.zeros(dim)) # to change

        if pooling not in ['mean', 'max', 'mean_sqrt']:
            raise ValueError("Pooling strategy isn't correct")

        # Perform mean-pooling
        if pooling == 'mean':
            text_embedding = np.vstack(embeddings).mean(axis=0)
        # Use max in each dimension over all tokens.
        if pooling == 'max':
            text_embedding = np.vstack(embeddings).max(axis=0)
        # Perform mean-pooling, but devide by sqrt(input_length).
        if pooling == 'mean_sqrt':
            text_embedding = np.vstack(embeddings).sum(axis=0) / np.sqrt(len(embeddings))
        
        return text_embedding


    def _bert_embedding(self, text):
        return self.model.encode(text)
    

    def embed(self, text, pooling:str = 'mean'):
        # pooling may be deleted later -> not been used
        if text is None or isinstance(text, float):
            text = ''
        if self.model_name == 'Glove':
            embedding = self._w2v_embedding(text, pooling)
        if self.model_name == 'SBert':
            embedding = self._bert_embedding(text)
        return embedding
    
    def get_vector_dim(self):
        if self.model_name == 'Glove':
            return self.model.vector_size
        if self.model_name == 'SBert':
            return self.model.get_sentence_embedding_dimension()
        


# generate embedding and calculating similarity between two texts
def generate_embedding_matrix(list_texts, filtered_ixs, embedder, info:str):
    '''
    @param: list_texts: list of texts to be embedded, could be labels/descriptions
            filtered_ixs: list of indices of texts not to be embedded -> exact_match with another class
            embedder: the embedding model used to embed texts
    '''
    embed_matrix = []
    for i,c_text in enumerate(tqdm(list_texts, desc='generate embedding for '+info+' ontology classes')):
        if i not in filtered_ixs:
            embedding = embedder.embed(c_text)
            embed_matrix.append(embedding)
        else:
            embed_matrix.append(np.zeros(embedder.get_vector_dim()))
    return np.array(embed_matrix)


def similarity_evaluate(source, target, embedder, similarity, metrics='cos'):
    '''
    @param: source: List[str], list of texts(labels/desc) of source ontology classes
            target: List[str], list of texts(labels/desc) of target ontology classes
            embedder: the embedding model used to embed texts
            similarity: the similarity matrix after string matching
            metrics: the similarity metrics to be chosen: 'cos', 'l2', 'l1' 
    @return: similarity: matrix with shape (n_source_classes, n_target_classes) 
    '''
    non_zero_row_indices = similarity.getnnz(axis=1).nonzero()[0]
    non_zero_column_indices = np.where(similarity.getnnz(axis=0) > 0)[0]

    src_matrix = generate_embedding_matrix(source, non_zero_row_indices, embedder, 'source')
    tar_matrix = generate_embedding_matrix(target, non_zero_column_indices, embedder, 'target')

    if metrics == 'cos':
        similarity = cosine_similarity(src_matrix, tar_matrix) # (n_samples1, n_samples2)
    if metrics == 'l2':
        similarity = - euclidean_distances(src_matrix, tar_matrix) # L2 norm
    if metrics == 'l1':
        similarity = - manhattan_distances(src_matrix, tar_matrix) # L1 norm
    
    return similarity


def pool_simi(desc_simi, label_simi, pooling='max', weights=[]):
    '''
    pooling two simi_matrix
    '''
    if pooling == 'max':
        class_simi = np.maximum(desc_simi, label_simi)
    
    if pooling == 'weighted': # contain 'mean' pooling
        if not weights:
            raise ValueError('Weights should be given as input')
        class_simi = weights[0]*desc_simi + weights[1]*label_simi

    return class_simi

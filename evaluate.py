import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


def mrr_score(y_true, y_score):
    n_samples, n_labels = y_true.shape
    mrr_sum = 0.0
    
    for i in range(n_samples):
        labels = np.argsort(y_score[i])[::-1]
        rank = 0
        
        for j in range(n_labels):
            if y_true[i, labels[j]] == 1:
                rank = j + 1
                break
        
        if rank > 0:
            mrr_sum += 1.0 / rank
    
    mrr = mrr_sum / n_samples
    return mrr


def hit_at_k(y_true, y_score, k):
    """
     @param:y_true: binary 2D array of true labels, shape = [n_samples, n_labels]
            y_score: 2D array of predicted label scores, shape = [n_samples, n_labels]
            k: number of top predictions to consider
    """
    top_k_preds = np.argsort(y_score, axis=1)[:, -k:]
    true_labels = [np.where(y_true[i] == 1)[0] for i in range(y_true.shape[0])]
    # All true labels count in top_k predictions
    hit_at_k = [np.intersect1d(true_labels[i], top_k_preds[i]).shape[0] for i in range(y_true.shape[0])]
    
    return np.sum(hit_at_k) / len(y_true.nonzero()[0])


def score(y_true, y_score, metrics='mrr', k=None):
    '''
    @param: y_true is of shape (n_samples,n_labels)
            y_score is of shape (n_samples,n_labels)
    '''
    # y_true is of shape (n_samples,n_labels)
    if metrics == 'mrr':
        mrr = mrr_score(y_true, y_score)
        print("MRR score is: {}".format(mrr))

    # if metrics == 'lrap':
    #     lrap = label_ranking_average_precision_score(y_true, y_score)
    #     print("Label_ranking_average_precision_score is: {}".format(lrap))

    if metrics == 'hit':
        if k is None:
            raise ValueError("Hit@k metrics must specify k value")
        hits = hit_at_k(y_true, y_score, k)
        print("Hit@{} Score is: {}".format(k, hits))


def ground_truth_generation(df_map, simi_score, source_cids, target_cids):
    '''
    @param: df_map: DataFrame, mapping pairs['Class1_id', 'Class2_id']
            simi_score: 2D array of predicted mapping scores, shape = [n_samples, n_labels]
            source_cids: List[str], list of ids of source ontology classes
            target_cids: List[str], list of ids of target ontology classes
    '''
    tarc2idx = {}
    for ix, t in enumerate(target_cids):
        tarc2idx[t] = ix
    
    y_true = np.zeros(simi_score.shape)
    for i in range(simi_score.shape[0]):
        tar_cids = df_map.loc[df_map['Class1_id'] == source_cids[i], 'Class2_id'].values
        for t in tar_cids:
            y_true[i, tarc2idx[t]] = 1
    return y_true


def Precision_Recall_F1(y_true, y_pred):
    '''
    @param: y_true: List[tuple], gold_standard for classes alignment
            y_pred: List[tuple], predicted alignment results
    '''
    # deal with partial gold standard:ignore any predicted matches if neither of the classes 
    # in that pair is present as a true positive pair with another class in the gold standard
    gold_source_cls = set(item[0] for item in y_true)
    gold_target_cls = set(item[1] for item in y_true)

    y_pred_filtered = []
    for pair in y_pred:
        if pair[0] in gold_source_cls or pair[1] in gold_target_cls:
            y_pred_filtered.append(pair)
    
    tp, fp, fn = [], y_pred_filtered, []
    for refered_cell in y_true:
        if refered_cell in y_pred:
            tp.append(refered_cell)
            fp.remove(refered_cell)
        else:
            fn.append(refered_cell)
    
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    F1 = 2 * precision * recall / (precision + recall)
    return np.array([precision, recall, F1])
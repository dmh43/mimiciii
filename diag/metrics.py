import numpy as np

def metrics_at_k(rankings_to_judge, relevant_doc_ids, k=10):
  correct = 0
  num_relevant = 0
  num_rankings_considered = 0
  avg_precision_sum = 0
  ndcgs = []
  for ranking, relevant in zip(rankings_to_judge, relevant_doc_ids):
    if ranking is None: continue
    num_relevant_in_ranking = len(relevant)
    if num_relevant_in_ranking == 0: continue
    avg_correct = 0
    correct_in_ranking = 0
    dcg = 0
    idcg = 0
    for doc_rank, doc_id in enumerate(ranking[:k]):
      rel = doc_id in relevant
      correct += rel
      correct_in_ranking += rel
      precision_so_far = correct_in_ranking / (doc_rank + 1)
      avg_correct += rel * precision_so_far
      dcg += (2 ** rel - 1) / np.log2(doc_rank + 2)
    num_relevant += num_relevant_in_ranking
    avg_precision_sum += avg_correct / min(k, num_relevant_in_ranking)
    idcg += np.array([1.0/np.log2(rank + 2)
                      for rank in range(min(k, num_relevant_in_ranking))]).sum()
    ndcgs.append(dcg / idcg)
    num_rankings_considered += 1
  precision_k = correct / (k * num_rankings_considered)
  recall_k = correct / num_relevant
  ndcg = sum(ndcgs) / len(ndcgs)
  mean_avg_precision = avg_precision_sum / num_rankings_considered
  return {'precision': precision_k,
          'recall': recall_k,
          'ndcg': ndcg,
          'map': mean_avg_precision}

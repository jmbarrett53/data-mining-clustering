# Submit this file to Gradescope
from typing import Dict, List, Tuple
import math
from math import comb
from collections import Counter
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.

class Solution:

  def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
    Args:
      true_labels: list of true labels
      pred_labels: list of predicted labels
    Returns:
      A dictionary of (true_label, pred_label): count
    """
    # Possible values for labels
    classes = set(true_labels)
    classes = classes.union(set(pred_labels))
    matrix = [[0 for i in classes] for i in classes]

    for true, pred in zip(true_labels, pred_labels):
      matrix[true][pred] += 1

    ret_dict = {}
    for row_id, row in enumerate(matrix):
      for col_id, col in enumerate(row):
        if col != 0:
          ret_dict[(row_id, col_id)] = col

    return ret_dict

  def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the Jaccard index.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The Jaccard index. Do NOT round this value.
    """
    
    tp, fn, fp = 0, 0, 0
    row_sums = {}
    col_sums = {}

    confusion_matrix = self.confusion_matrix(true_labels=true_labels, pred_labels=pred_labels)
    # Calculate TP by summing combinations within each cell
    for (true_label, pred_label), count in confusion_matrix.items():
      tp += comb(count, 2)
      row_sums[pred_label] = row_sums.get(pred_label, 0) + count
      col_sums[true_label] = col_sums.get(true_label, 0) + count


    fp = sum(comb(n, 2) for n in row_sums.values()) - tp
    fn = sum(comb(n, 2) for n in col_sums.values()) - tp
    

    jaccard = tp / (tp + fn + fp)

    return jaccard


        
    

  def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the normalized mutual information.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The normalized mutual information. Do NOT round this value.
    """
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    total = len(true_labels)


    H_true, H_pred = 0, 0
    # Calculate entropy
    for count in true_counts.values():
      H_true += -(count / total) * math.log(count / total)

    for count in pred_counts.values():  
      H_pred += -(count / total) * math.log(count / total)

    # Calculate joint entropy
    joint_counts = Counter(zip(true_labels, pred_labels))


    I_true_pred = 0
    for (true, pred), count in joint_counts.items():
      I_true_pred += (count / total) * math.log((count * total) / (true_counts[true] * pred_counts[pred]))

    if (H_true + H_pred):
      return (2 * I_true_pred) / (H_true + H_pred)
    else:
      return 0.0



# test_1_true = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
# test_1_pred = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]

# test_2_true = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
# test_2_pred = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]

# test = Solution()
# print(test.confusion_matrix(test_2_true, test_2_pred))

# print(test.jaccard(test_2_true, test_2_pred))

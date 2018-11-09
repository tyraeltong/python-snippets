import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_labels, normalize=False, figsize=(10, 7), fontsize=14):
  """Prints a confusion matrix returned by sklearn.metrics.confusion_matrix as a heatmap.

  Arguments
    confusion_matrix: numpy.ndarray
      returned from a call to sklearn.metrics.confusion_matrix.
    class_labels: list
      an ordered list of class names in the same order as confusion matrix.
    normalize: boolean
      True to normalize confusion matrix to 0.0 - 1.0
    figsize: tuple
      plot figsize.
    fontsize: int
      for axes labels.
  Returns
    matplotlib.figure.Figure
      The heatmap figure generated
  """
  if normalize:
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confusion matrix without normalization')

  df = pd.DataFrame(confusion_matrix, index=class_labels, columns=class_labels)
  fmt = '.2f' if normalize else 'd'
  fig = plt.figure(figsize=figsize)
  try:
    heatmap = sns.heatmap(df, annot=True, fmt=fmt)
  except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
  
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return fig
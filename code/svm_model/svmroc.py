print(__doc__)
## Generate from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
import sys
import csv
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def load_file(file_path,i):
    sentiments = []
    texts = []
    with open(file_path, 'r') as file_reader:
        reader = csv.reader(file_reader)
        for row in reader:
            # print row
            # sentiment = [row[6],row[7]]
            sentiment = [row[6],row[7],row[8],row[10],row[11]]
            sentiment = sentiment[:i]
            text = row[3]
            sentiments.append(sentiment)
            texts.append(text)
    return normalize(sentiments), np.array(texts)
def load_file1(file_path):
    sentiments = []
    texts = []
    with open(file_path, 'r') as file_reader:
        reader = csv.reader(file_reader)
        for row in reader:
            # print row
            sentiment = [row[6],row[7]]
            text = row[3]
            sentiments.append(sentiment)
            texts.append(text)
    return normalize(sentiments), np.array(texts)

def normalize(X) :
  X = np.array(X)
  return (preprocessing.scale(X))
def transformtolabel(x,classname):
    y = []
    x = map(float,x)
    ave = sum(x)/len(x)
    for i in range(0,len(x)):
        if float(x[i]) > ave:
            # print x[i]
            y.append([1,0])
        else: y.append([0,1])
    return y,ave





def main():
  plt.figure()
  for j in range(1,6):
    random_state = np.random.RandomState(0)
    X,y = load_file(file_name,j)
    k = 2
    # y = label_binarize(y, classes=[0, 1, 2])
    # n_classes = y.shape[1]
    # print n_classes
    n_classes = 2
    ylabel, ave= transformtolabel(y,k)
    ylabel = np.array(ylabel)
    # ylabel = np.transpose(ylabel)
  # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, ylabel, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      # print y_test[i]
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print fpr[1]

    ##############################################################################

    # Plot of a ROC curve for a specific class

    # plt.figure()
    # plt.plot(fpr[0], tpr[0], label='CO below %0.2f' % ave +' (area = %0.2f)' %roc_auc[0])
    plt.plot(fpr[1], tpr[1], label='O3 prediction (area = %0.2f)' %roc_auc[1]+'(%0.0f'% j+' features)')
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic for SVM')
  plt.legend(loc="lower right")
  plt.show()


  
if __name__ == '__main__':
  file_name = sys.argv[1]
  main()
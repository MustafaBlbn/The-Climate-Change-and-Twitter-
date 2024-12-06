import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df2=pd.read_csv('The Climate Change Twitter Dataset.csv')

silinecek_sutunlar = ['temperature_avg', 'id', 'lat', 'lng']
df2 = df2.drop(silinecek_sutunlar, axis=1)
df2.loc[df2['aggressiveness'] == 'aggressi', 'aggressiveness'] = 'aggressive'
df2['created_at'] = pd.to_datetime(df2['created_at']).dt.date
df2.head(10)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

basliklar = df2.select_dtypes (include = "object"). columns

for label in basliklar:
  df2 [label] =le.fit_transform(df2[label].astype(str))


yeni=df2.head(300)

X = yeni.iloc[:, :-1].values
Y = yeni.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

nb = GaussianNB()
logistic_reg = LogisticRegression(random_state = 42)
knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
kernel_svm = SVC(kernel = 'linear', random_state = 42)
decision_tree = DecisionTreeClassifier (criterion = 'entropy', random_state = 0)
random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
models = ["Naive Bayes", "Logistic Reg", "KNN", "Kernel SVM", "Decision Tree", "Random Forest"]

def train_models(classifier, X_train, Y_train):
   classifier.fit(X_train, Y_train)

def test_models(classifier, X_test):
   return classifier.predict(X_test)

classifier = [nb, logistic_reg, knn, kernel_svm, decision_tree, random_forest]
Y_pred_val = [0]*6
for i in range(6):
  train_models(classifier[i], X_train, Y_train)
  Y_pred = test_models(classifier[i], X_test)
  Y_pred_val[i] = Y_pred

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
model_accuracy = [0]*6
def create_confusion_matrix(Y_test, Y_pred, models) :
  for m in range(len(Y_pred)):
    cm = confusion_matrix(Y_test, Y_pred[m])
    figure, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.matshow(cm, cmap=plt.cm.Wistia, alpha=0.5)
    for i in range(cm.shape[0]): 
      for j in range(cm. shape [1]):
        ax. text(x=j, y=i,
          s= cm[i,j],
          va= 'center', ha='center')
    plt.xlabel ('Predicted Label') 
    plt.ylabel('True Label')
    plt.title(f"{models [m]}")
    plt.show()

    accuracy = accuracy_score(Y_test, Y_pred[m])
    class_report = classification_report(Y_test, Y_pred [m])
    print(f"Accuracy {accuracy:.2f}\n")
    print(f"{class_report}\n")
    model_accuracy[m] = accuracy


create_confusion_matrix(Y_test, Y_pred_val, models)

Y_pred = logistic_reg.predict(X_test)
from sklearn import metrics
fpr, tpr, metrics.roc_curve(Y_test, Y_pred)

plt.plot(fpr, tpr, 'm')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Gender Classifier')
plt.xlabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.grid(True)

print(metrics.roc_auc_score(Y_test, Y_pred))

from sklearn import tree
plt.figure(figsize=(40,30))
tree.plot_tree(decision_tree, filled=True, fontsize=20)

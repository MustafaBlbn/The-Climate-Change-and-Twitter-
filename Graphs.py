import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('The Climate Change Twitter Dataset.csv')
df2 = pd.read_csv('disasters.csv')
df.head()
df.tail(10)
df.info()
silinecek_sutunlar = ['temperature_avg', 'id', 'lat', 'lon']
df = df.drop(silinecek_sutunlar, axis=1)
df.loc[df['aggressiveness'] == 'aggressi', 'aggressiveness'] = 'aggressive'
df['created_at'] = pd.to_datetime(df['created_at']).dt.date
df.head(10)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
basliklar = df.select_dtypes (include = "object"). columns
for label in basliklar:
  df[label] =le.fit_transform(df[label].astype(str))

df.head(20)

plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict= {'fontsize' : 12}, pad = 12)
plt.show()


figsize = (12, 1.2 * len(df['aggressiveness'].unique()))
plt.figure(figsize=figsize)

df['gender_code'] = df['gender'].map({'male': 1, 'female': 2, 'undefined': 3})

filtered_df = df[df['gender_code'].isin([1, 2, 3])]

sns.violinplot(data=filtered_df, x='gender_code', y='aggressiveness', inner='box', palette='Dark2', 
               legend=False, width=0.8, dodge=True)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.xticks(ticks=[0, 1, 2], labels=['male', 'female', 'undefined'])

plt.show()

figsize = (12, 1.2 * len(df['gender'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(df, x='sentiment', y='gender', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()

figsize = (12, 1.2 * len(df['stance'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(df, x='sentiment', y='stance', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()

topic_counts = df['topic'].value_counts()

topics = topic_counts.index
tweet_counts = topic_counts.values

plt.figure(figsize=(12, 6))
plt.plot(topics, tweet_counts, marker='o', linestyle='-')
plt.title('Toplam Tweet Sayısı - Her Başlık için')
plt.xlabel('Başlıklar')
plt.ylabel('Tweet Sayısı')
plt.xticks(rotation=90)  # Başlıkları dikey olarak yazdırın
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(df, x='lng', y='aggressiveness', inner='box', palette='Dark2', ax=axs[0])
axs[0].set_title('Lng - Aggressiveness')

sns.violinplot(df, x='lat', y='aggressiveness', inner='box', palette='Dark2', ax=axs[1])
axs[1].set_title('Lat - Aggressiveness')

for ax in axs:
    sns.despine(ax=ax, top=True, right=True, bottom=True, left=True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df[df['aggressiveness'] == 'not aggressive'], x='lng', y='lat', color='blue', 
                label='Not Aggressive')
sns.scatterplot(data=df[df['aggressiveness'] == 'aggressive'], x='lng', y='lat', color='red', 
                label='Aggressive')

plt.xlabel('Longitude (Lng)')
plt.ylabel('Latitude (Lat)')
plt.title('Aggressiveness across Longitude and Latitude')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(data=df, x='lng', y='aggressiveness', inner='box', palette='Dark2', ax=axs[0])
axs[0].set_title('Lng - Aggressiveness')
sns.despine(ax=axs[0], top=True, right=True, bottom=True, left=True)

sns.violinplot(data=df, x='lat', y='aggressiveness', inner='box', palette='Dark2', ax=axs[1])
axs[1].set_title('Lat - Aggressiveness')
sns.despine(ax=axs[1], top=True, right=True, bottom=True, left=True)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(data=df, x='lng', y='stance', inner='box', palette='Dark2', ax=axs[0])
axs[0].set_title('Lng - Stance')
sns.despine(ax=axs[0], top=True, right=True, bottom=True, left=True)

sns.violinplot(data=df, x='lat', y='stance', inner='box', palette='Dark2', ax=axs[1])
axs[1].set_title('Lat - Stance')
sns.despine(ax=axs[1], top=True, right=True, bottom=True, left=True)

plt.tight_layout()
plt.show()

from matplotlib import pyplot as plt
df['sentiment'].plot(kind='kde', figsize=(8, 4), title='sentiment')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

df['sentiment'].plot(kind='hist', bins=20, title='sentiment')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

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

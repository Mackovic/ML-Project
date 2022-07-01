import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model  import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

models = {}
models['Logistic Regression'] = LogisticRegression(max_iter=1000)
models['Support Vector Machines'] = LinearSVC(dual=False)
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Naive Bayes'] = GaussianNB()
models['K-Nearest Neighbor'] = KNeighborsClassifier()

accuracy, precision, recall = {}, {}, {}

data = pd.read_csv("DelayedFlights.csv")

cancelled_cols = ["Month", "DayofMonth", "DayOfWeek",
                  "DepTime", "DepDelay" , "Distance"]

cancelled_flights = data.query("Cancelled == 1")
not_cancelled_flights = data.query("Cancelled == 0").iloc[:10000]

fixed_data = pd.concat([cancelled_flights, not_cancelled_flights]).sample(frac=1).reset_index(drop=True)

X = fixed_data[cancelled_cols]
y = fixed_data.Cancelled

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

for key in models.keys():
    accuracy[key]= cross_val_score(models[key], X, y, scoring="accuracy").mean()
    precision[key]= cross_val_score(models[key], X, y, scoring="precision").mean()
    recall[key]= cross_val_score(models[key], X, y, scoring="recall").mean()


df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)
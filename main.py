import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.special import expit
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pandas as pd
import arff
import numpy as np

# Loading Data
with open('./medical-train.arff', 'r') as file:
    arff_data = arff.load(file)

data = pd.DataFrame(arff_data['data'])
data.columns = [attribute[0] for attribute in arff_data['attributes']]

# Filtering and Pre-Processing
for col in data.columns:
    if col.startswith('Class'):
        data[col] = data[col].apply(lambda x: 1 if x == '1' else 0)

X = data.drop(columns=[col for col in data.columns if col.startswith('Class')])
y = data[[col for col in data.columns if col.startswith('Class')]]

sum_x = y.sum(axis=1) == 1
X = X[~sum_x]
y = y[~sum_x]

print("X:", X.shape)
print("Y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filtering single label data
single_class_columns = [col for col in y_train.columns if len(y_train[col].unique()) == 1]
print("Columns with a single class in the training set:", single_class_columns)

y_train = y_train.drop(columns=single_class_columns)
y_test = y_test.drop(columns=single_class_columns)

t = 5
s = 3

training_examples = []
micro_f1_scores = []

# Main Algorithm
for i in range(t):

    print(f"Iteration {i+1}\n")
    # Training Binary Classifier using SVM
    svc = SVC(kernel='linear', probability=True)
    multi_target_svc = MultiOutputClassifier(svc)
    multi_target_svc.fit(X_train, y_train)

    # LR-Based Prediction Method
    decision_values = np.zeros((X_train.shape[0], y_train.shape[1]))
    for i, estimator in enumerate(multi_target_svc.estimators_):
        decision_values[:, i] = estimator.decision_function(X_train)

    y_prob = expit(decision_values)
    y_prob_norm = normalize(y_prob, norm='l1', axis=1)
    lr_features = np.zeros_like(y_prob_norm)
    for i in range(y_prob_norm.shape[0]):
        sorted_indices = np.argsort(-y_prob_norm[i])
        lr_features[i, :] = y_prob_norm[i, sorted_indices]

    num_labels = y_train.sum(axis=1).values
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
    print(lr_features.shape)
    lr.fit(lr_features[:X_train.shape[0]], num_labels)

    pred_labels = lr.predict(lr_features)

    scores = np.zeros(X_test.shape[0])
    for j in range(X_test.shape[0]):
        top_label = np.argsort(-y_prob_norm[j])[:pred_labels[j]]
        y_pred = np.zeros(y_train.shape[1])
        y_pred[top_label] = 1

        scores[j] = np.sum((1 - y_pred * decision_values[j]) / 2)

    top_indices = np.argsort(scores)[-s:]

    x_sel = X_test.iloc[top_indices]
    y_sel = y_test.iloc[top_indices]

    X_train = pd.concat([X_train, x_sel])
    y_train = pd.concat([y_train, y_sel])

    X_test = X_test.drop(X_test.index[top_indices])
    y_test = y_test.drop(y_test.index[top_indices])

    # Append training examples and micro F1-score
    training_examples.append(X_train.shape[0])
    y_pred_train = multi_target_svc.predict(X_train)
    f1 = f1_score(y_train, y_pred_train, average='micro')
    micro_f1_scores.append(f1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

single_class_columns = [col for col in y_train.columns if len(y_train[col].unique()) == 1]
print("Columns with a single class in the training set:", single_class_columns)

y_train = y_train.drop(columns=single_class_columns)
y_test = y_test.drop(columns=single_class_columns)
# Train final model and evaluate
multi_target_svc.fit(X_train, y_train)
y_pred = multi_target_svc.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot the micro F1-score against the number of training examples
plt.plot(training_examples, micro_f1_scores, marker='o')
plt.xlabel('Number of Training Examples')
plt.ylabel('Micro F1-score')
plt.title('Micro F1-score vs Number of Training Examples')
plt.grid(True)
# plt.ylim(0, 1)  # Set y-scale to be between 0 and 1
plt.show()


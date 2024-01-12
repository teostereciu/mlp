import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from asl_alphabet_recognizer.data.preprocessing import DataHandler
from config.settings import data_dir, models_dir, debug
import argparse

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns


def show_confusion_matrix(clf, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# parser = argparse.ArgumentParser(description='ASL alphabet recognizer project.')
# parser.add_argument('--skip_preprocess', type=bool, default=False, help='Whether to skip preprocessing')
# parser.add_argument('--skip_training', type=str, default=False, help='Whether to skip training')

# args = parser.parse_args()

data_handler = DataHandler(data_dir)
(data_handler.unzip()
 .create_filename_dataframe()
 .sample(n=100)
 .process()
 .apply_canny())

df_train, df_test = data_handler.get_dfs()
# data_handler.save_dfs()
# df_train_saved, df_test_saved = data_handler.load_dfs()

X_train = np.array(df_train['X_canny'].apply(lambda arr: np.array(arr).astype(int)).tolist())
y_train = np.array(df_train['category'].tolist())
X_test = np.array(df_test['X_canny'].apply(lambda arr: np.array(arr).astype(int)).tolist())
y_test = np.array(df_test['category'].tolist())

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

if debug:
    data_handler.show_before_and_after()
    print(df_train.head())
    print(df_train.shape)
    print(df_test.shape)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    print("X_train_flat shape: ", X_train_flat.shape)
    print("X_test_flat shape: ", X_test_flat.shape)

# create and train baseline model
clf = svm.SVC()
clf.fit(X_train_flat, y_train)

# results

y_pred = clf.predict(X_train_flat)
accuracy = accuracy_score(y_train, y_pred)
print(f"Train Accuracy: {accuracy:.2%}")
y_pred = clf.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")

show_confusion_matrix(clf, y_test, y_pred)

report = classification_report(y_test, y_pred)
print(report)

# save model
with open(os.path.join(models_dir, 'baseline.pkl'),'wb') as f:
    pickle.dump(clf, f)

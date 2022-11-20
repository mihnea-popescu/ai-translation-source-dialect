import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
import time

data_path = '.'
train_data_df = pd.read_csv(os.path.join(data_path, '../train_data.csv'))
test_data_df = pd.read_csv(os.path.join(data_path, '../test_data.csv'))

etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta] = idx
    id2label[idx] = eticheta

labels = train_data_df['label'].apply(lambda etich: label2id[etich])

data = train_data_df['text']


# Using scikit train test split to split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

print("Nr de date de antrenare: ", len(X_train))
print("Nr de date de testare: ", len(X_test))


# Bag of Words
cv = CountVectorizer()

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


# Testing
model = svm.LinearSVC(C=0.005)

model.fit(X_train, y_train)
tpreds = model.predict(X_test)
print('Acuratete pe datele de test: ', accuracy_score(tpreds, y_test))


# Vectorize the whole dataset
X_all = cv.fit_transform(data)


# 5-fold Cross Validation
scores = cross_val_score(model, X_all, train_data_df['label'], cv=5)
print("5-fold Cross Validation: acuratețe %0.2f cu abatere standard %0.2f" % (scores.mean(), scores.std()))


# Train on entire dataset
start_time = time.time()
model.fit(X_all, train_data_df['label'])
print('Secunde pentru antrenare: ', (time.time() - start_time))

train_predictions = model.predict(X_all)
print('Acuratete pe datele deja vazute de model: ', accuracy_score(train_data_df['label'], train_predictions))

# Confusion Matrix
cm = confusion_matrix(train_data_df['label'], train_predictions, labels=train_data_df['label'].unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label2id)
disp.plot()
plt.xlabel('Predictii')
plt.ylabel('Valori reale')
plt.title('Matrice de confuzie')
plt.rc('image', cmap='Greens')
plt.show()


# Process Data from Kaggle
X_kaggle = cv.transform(test_data_df['text'])
predictions = model.predict(X_kaggle)

rezultat = pd.DataFrame({'id': np.arange(1, len(predictions)+1), 'label': predictions})
print(rezultat)


# Save data in csv for kaggle submission
nume_model = str(model)
print(nume_model)
nr_de_caracteristici = 'N:NO_LIMIT'
print(nr_de_caracteristici)
functie_preprocesare = 'count_vectorizer_lower_split_25'
print(functie_preprocesare)

nume_fisier = '_'.join([nume_model, nr_de_caracteristici, functie_preprocesare]) + '.csv'
print('Nume experiment: ', nume_fisier)

rezultat.to_csv(nume_fisier, index=False)

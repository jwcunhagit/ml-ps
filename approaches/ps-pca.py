#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
train_ds = pd.read_csv('dataset/train.csv')
#test_ds = pd.read_csv('dataset/test.csv')

X = train_ds.iloc[:, 2:58].values
y = train_ds.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


X_train.shape
X_test.shape

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=2, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)




### ----
### save and retrieve model
### reference: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# serialize model to JSON
model_json = classifier.to_json()
with open("model-v2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("ann_v2.h5")
print("Saved model to disk")


# ----- load model from json ------

from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ann_2r_1s_adam_bin_crossentropy.h5")
print("Loaded model from disk")


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
X_pred = test_ds.iloc[:,1:57].values
y_pred = classifier.predict(X_pred)[:,0]
#y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
#model_vars = [a for a in test_ds.columns if 'id' not in a and 'target' not in a]

y_pred = loaded_model.predict_proba(test_ds[X_pred])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

sub = pd.DataFrame({'id': test_ds.id, 'target': y_pred[:,]})
sub.head()


sub.to_csv('output/ann_v2.csv', index=False, header=True)
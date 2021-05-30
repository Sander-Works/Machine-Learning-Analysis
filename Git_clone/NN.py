import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import datasets, model_selection
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
import warnings

warnings.filterwarnings("ignore")

# Loading dataset
dataset = datasets.load_breast_cancer()

# Putting the dataset into two arrays.
x = dataset.data
y = dataset.target

# variable of epochs
epochs = 100

# Splitting the arrays into random train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Scaling the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Looping to find the optimal numbers of neurons in each layer
L_range = range(1, 20)
scores = []
for l in L_range:
    mlp = MLPClassifier(hidden_layer_sizes=(l, l, l), max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)
    scores.append(mlp.score(X_test, y_test))

# Plotting accuracy for each set of neurons
plt.suptitle('MLP')
plt.xlabel('Nodes')
plt.ylabel('accuracy')
plt.scatter(L_range, scores)
plt.plot(L_range, scores)
plt.xticks([0, 5, 10, 15, 21])
plt.show()

# initializing MLP with the optimal numbers of neurons
mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

# Evaluating model cross val score with 10-kfold
kfold = model_selection.KFold(n_splits=10)
cv_results = model_selection.cross_val_score(mlp, X_train, y_train, cv=kfold)

# Printing results
print(classification_report(y_test, predictions))
print('MLP Accuracy: {:.2f}%'.format(accuracy_score(y_test, predictions) * 100))
print("MLP training set score: %f" % mlp.score(X_train, y_train))
print("MLP test set score: %f" % mlp.score(X_test, y_test))

print("\nThis will take minute")

# initializing learning_curve for MLP
train_sizes, train_scores, test_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=(4, 4, 4), max_iter=1000, random_state=1),
    x, y, cv=10,
    scoring='accuracy',
    n_jobs=1,
    # 20 different sizes of the training set
    train_sizes=np.linspace(0.01, 1, 20))

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plotting learning curve
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.title("MLP learning Curve")
plt.xlabel("Training size")
plt.ylabel('Accuracy curve')
plt.legend(loc='best')
plt.show()

# CNN with Keras
X_train = X_train.reshape(455, 30, 1)
X_test = X_test.reshape(114, 30, 1)
CNN_model = Sequential()
CNN_model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30, 1)))
CNN_model.add(BatchNormalization())
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(Dropout(0.5))

CNN_model.add(Flatten())
CNN_model.add(Dense(64, activation='relu'))
CNN_model.add(Dropout(0.5))

CNN_model.add(Dense(1, activation='sigmoid'))

# Defining the optimizer, the loss function and metrics.
CNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model for the fixed number of epochs
CNN_training_history = CNN_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
CNN_evaluation = CNN_model.evaluate(X_test, y_test, verbose=1)

print('CNN test loss: ', CNN_evaluation[0])
print('CNN test accuracy: ', CNN_evaluation[1])

epoch_range = range(1, epochs + 1)

CNN_accuracy = CNN_training_history.history['accuracy']
CNN_val_accuracy = CNN_training_history.history['val_accuracy']
CNN_loss = CNN_training_history.history['loss']
CNN_val_loss = CNN_training_history.history['val_loss']

epochs_range = range(len(CNN_accuracy))

# Plot training & validation accuracy values
CNN_model.summary()
plt.plot(epoch_range, CNN_accuracy)
plt.plot(epoch_range, CNN_val_accuracy)

# Plotting CNN model accuracy for each epoch
plt.title('CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.title('CNN Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(epoch_range, CNN_loss)
plt.plot(epoch_range, CNN_val_loss)
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

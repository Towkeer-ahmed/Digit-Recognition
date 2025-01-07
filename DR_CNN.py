#Digit recognition using CNN
#Data collection and preprocessing the dataset
#Using MNIST dataset

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import seaborn as sns 

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Print the shape of the dataset
print("Training data shape:", X_train.shape)  
print("Training labels shape:", Y_train.shape)  

# Plot the first 5 images in the train dataset along with their labels
plt.figure(figsize=(10, 2))  
for i in range(5):
    plt.subplot(1, 5, i + 1)  
    plt.imshow(X_train[i], cmap='gray')  # Plot the image in grayscale
    plt.title("Label: " + str(Y_train[i]))  
    plt.axis('off')  
plt.show()

# Plot the first 5 images in the test dataset along with their labels
plt.figure(figsize=(10, 2)) 
for i in range(5):
    plt.subplot(1, 5, i + 1) 
    plt.imshow(X_test[i], cmap='gray')  # Plot the image in grayscale
    plt.title("Label: " + str(Y_test[i]))  
    plt.axis('off')  
plt.show()

# Normalize the input data
X_train_normalized=X_train / 255.0
#print("X_train normalized\n", X_train_normalized[0, :])
X_test_normalized = X_test / 255.0
#print("X_test normalized\n", X_train_normalized[0, :])

# Reshape the data to fit the model
X_train = X_train_normalized.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test_normalized.reshape((X_test.shape[0], 28, 28, 1))

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Build the CNN model
model_CNN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model_CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Train the model
Result = model_CNN.fit(X_train, Y_train, epochs=10, validation_split=0.1)


# Evaluate the model and print metrics
test_loss, test_acc, test_precision, test_recall = model_CNN.evaluate(X_test, Y_test, verbose=2)
print('\nTest accuracy: {:.2f}%'.format(test_acc * 100))
print('Test loss: {:.2f}%'.format(test_loss * 100))
print('Precision: {:.2f}%'.format(test_precision * 100))
print('Recall: {:.2f}%'.format(test_recall * 100))

# Predict the labels
predictions = model_CNN.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(Y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix using CNN')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(Result.history['accuracy'])
plt.plot(Result.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')





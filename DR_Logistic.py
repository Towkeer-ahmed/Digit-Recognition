#Digit recognition using logistic regression
#Data collection and preprocessing the dataset
#Using MNIST dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns


train_df=pd.read_csv("mnist_train.csv")
test_df=pd.read_csv("mnist_test.csv")
print(train_df)
print(test_df)
train_array=np.array(train_df)
test_array=np.array(test_df)

#Shuffle the data
np.random.shuffle(train_array)
np.random.shuffle(test_array)


X_train, Y_train=train_array[:, 1:], train_array[:, 0]

X_test, Y_test=test_array[:, 1:], test_array[:,0]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# Visualize the first 5 digits from the training set
plt.figure(figsize=(10, 2))
for idx in range(5):
    plt.subplot(1, 5, idx + 1)
    plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {Y_train[idx]}")
    plt.axis('off')

plt.show()

# Visualize the first 5 digits from the training set
plt.figure(figsize=(10, 2))
for idx in range(5):
    plt.subplot(1, 5, idx + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {Y_test[idx]}")
    plt.axis('off')

plt.show()


# Normalize the data
X_train_normalized=X_train/255
print("X_train normalized\n", X_train_normalized[0, :])

X_test_normalized=X_test/255
print("X_test normalized\n", X_test_normalized[0, :])

#Train the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train_normalized, Y_train)

#predict on the train set
train_pred=model.predict(X_train_normalized)

#predict on the test set, used softtmax fucntion
test_pred= model.predict(X_test_normalized)
#print("Test prediction", test_pred)

#Evaluation'of the model
#Training accuracy and testing accuracy, precision, recall

train_acc=accuracy_score(Y_train, train_pred)
test_acc=accuracy_score(Y_test, test_pred)
precision = precision_score(Y_test, test_pred, average='macro') 
recall = recall_score(Y_test, test_pred, average='macro')

print("Training accuracy", round(train_acc*100, 4))
print("Accuracy", round(test_acc*100, 4))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))


#Confusion matrix
cm=confusion_matrix(Y_test, test_pred)

plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, cmap="Blues_r", fmt='0.4g')
plt.xlabel("predicted values")
plt.ylabel("Actual values")
plt.title('Confusion Matrix using Logistic Regression')
plt.show()







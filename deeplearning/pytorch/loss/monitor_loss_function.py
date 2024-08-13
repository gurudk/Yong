from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2
)

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


class PyTorch_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyTorch_NN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x


def get_accuracy(pred_arr, original_arr):
    pred_arr = pred_arr.detach().numpy()
    original_arr = original_arr.numpy()
    final_pred = []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count += 1
    return count / len(final_pred) * 100


def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs):
    train_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):

        # forward feed
        output_train = model(X_train)

        train_accuracy.append(get_accuracy(output_train, y_train))

        # calculate the loss
        loss = criterion(output_train, y_train)
        train_loss.append(loss.item())

        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # backward propagation: calculate gradients
        loss.backward()

        # update the weights
        optimizer.step()

        with torch.no_grad():
            output_test = model(X_test)
            test_accuracy.append(get_accuracy(output_test, y_test))

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy) / len(train_accuracy):.2f}, Test Accuracy: {sum(test_accuracy) / len(test_accuracy):.2f}")

    return train_loss, train_accuracy, test_accuracy


input_dim = 4
output_dim = 3
learning_rate = 0.01

model = PyTorch_NN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_loss, train_accuracy, test_accuracy = train_network(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    num_epochs=100,
)

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6), sharex=True)

ax1.plot(train_accuracy)
ax1.set_ylabel("training accuracy")

ax2.plot(train_loss)
ax2.set_ylabel("training loss")

ax3.plot(test_accuracy)
ax3.set_ylabel("test accuracy")

ax3.set_xlabel("epochs")

plt.show()

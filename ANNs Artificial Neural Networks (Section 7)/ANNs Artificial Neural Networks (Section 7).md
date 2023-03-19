## Loss function and cost function concept (ANN math part 2 (errors, loss, cost): Video 43)

**Details of loss functions and cost functions can be found on Andrew NG: Machine Learning Specialization**

The loss function measures the error from one sample. 

Loss functions for regression and classification:

<img src="./Images/1.png" title="" alt="" width="821">

the cost function is the average of all the loss functions by number of training samples. 

## Construct ANN for simple regression on Pytorch (ANN for regression: Video 45)

Jupyter notebook:

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```

```python
# creating the dataset
N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1) / 2
fig, ax = plt.subplots()
ax.plot(x, y, 's')
plt.show()
```

<img src="./Images/2.png" title="" alt="" width="565">

```python
ANNreg = nn.Sequential(
    nn.Linear(1, 1), # input layer, (1,1) takes 1 input, sends 1 output
    nn.ReLU(), # activation function
    nn.Linear(1, 1) # output layer
)
ANNreg
```

Sequential(
  (0): Linear(in_features=1, out_features=1, bias=True)
  (1): ReLU()
  (2): Linear(in_features=1, out_features=1, bias=True)
)

```python
learning_rate = 0.05
loss_func = nn.MSELoss()
# type of gradient descent
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate) 
```

```python
num_epochs = 500
losses = torch.zeros(num_epochs)

# training
for epochi in range(num_epochs):
    y_hat = ANNreg(x) # forward pass
    loss = loss_func(y_hat, y) # compute loss
    losses[epochi] = loss
    optimizer.zero_grad() # back prop
    loss.backward()
    optimizer.step()
```

The cell below shows what should happen ideally, as loss is decreasing on each successive epoch

```python
# manually computing loss
# final forward pass
preds = ANNreg(x)
# final loss (MSE)
test_loss = (preds - y).pow(2).mean()

# plotting epoch vs loss
fig, ax = plt.subplots()
ax.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
ax.plot(num_epochs, test_loss.detach(), 'ro')
ax.set(xlabel="Epoch", ylabel="Loss", title=f"Final Loss {test_loss.item()}")
plt.show()
```

<img src="./Images/3.png" title="" alt="" width="505">

To understand more about the .detach() and item() used above, look at the comparison below:

```python
test_loss, test_loss.detach(), test_loss.item()
```

(tensor(0.1770, grad_fn=<MeanBackward0>), tensor(0.1770), 0.1769895702600479)

Now, to know how the trained model predicts the data:

```python
fig, ax = plt.subplots()
ax.plot(x, y, 'bo', label='Real data')
ax.plot(x, preds.detach(), 'rs', label="Predictions")
plt.title(f'prediction-data r={np.corrcoef(y.T, preds.detach().T)[0,1]:.2f}')
plt.legend()
plt.show()
```

<img src="./Images/4.png" title="" alt="" width="465">

## Experimenting how accuracy/loss changes with slope variation (CodeChallenge: manipulate regression slopes: Video 46)

Jupyter notebook (continuation):

```python
def create_the_data(m):
    N = 50
    x = torch.randn(N, 1)
    y = m * x + torch.randn(N, 1) / 2
    return x, y
```

```python
def build_and_train_model(x, y):
    ANNreg = nn.Sequential(
        nn.Linear(1, 1), # input layer, (1,1) takes 1 input, sends 1 output
        nn.ReLU(), # activation function
        nn.Linear(1, 1) # output layer
    )
    learning_rate = 0.05
    loss_func = nn.MSELoss()
    # type of gradient descent
    optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)

    num_epochs = 500
    losses = torch.zeros(num_epochs)

    # training
    for epochi in range(num_epochs):
        y_hat = ANNreg(x) # forward pass
        loss = loss_func(y_hat, y) # compute loss
        losses[epochi] = loss
        optimizer.zero_grad() # back prop
        loss.backward()
        optimizer.step()
    preds = ANNreg(x)
    return preds, losses
```

The cell below is just to test out the functions above

```python
x,y = create_the_data(0.8)
y_hat, losses = build_and_train_model(x, y)

fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,4))
ax0.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=0.1)
ax.set(xlabel='Epoch', title='Loss')
ax1.plot(x, y, 'bo', label='Real data')
ax1.plot(x, y_hat.detach(), 'rs', label='Predictions')
ax1.set(xlabel='x', ylabel='y', title=f'prediction-data corr={np.corrcoef(y.T, y_hat.detach().T)[0,1]:.2f}')
ax1.legend()
plt.show()
```

<img src="./Images/5.png" title="" alt="" width="612">

```python
slopes = np.linspace(-2,2,21)
# repeat experiment 50 times
num_exps = 50
results = np.zeros((len(slopes),num_exps,2))
for slopei in range(len(slopes)):
    for N in range(num_exps):
        x,y = create_the_data(slopes[slopei])
        y_hat, losses = build_and_train_model(x, y)
        # store loss from final step of training and performance for plotting graph
        results[slopei, N, 0] = losses[-1]
        results[slopei, N, 1] = np.corrcoef(y.T, y_hat.detach().T)[0,1]
results[np.isnan(results)] = 0
```

```python
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,4))
ax0.plot(slopes, np.mean(results[:,:,0], axis=1), 'ko-', markerfacecolor='w', markersize=10)
ax0.set(xlabel='Scope', title='Loss')

ax1.plot(slopes, np.mean(results[:,:,1],axis=1), 'ms-', markerfacecolor='w', markersize=10)
ax1.set(xlabel='Scope', ylabel='Real predicted correlation', title='Model Performance')
plt.show()
```

<img src="./Images/6.png" title="" alt="" width="855">

On the graph above:

Why were losses larger with larger slopes, even though the fit to the data was better?

* Losses are not normalized; they are in the scale of the data. Larger slopes led to more variance in y, and so the losses are larger as the data values are larger.

Why did model accuracy drop when the slopes were closer to zero?

* x is less informative about y when the slope decreases. The model had less information about y.

## ANN on classification pytorch (ANN for classifying qwerties: Video 47)

This section explores ANN's on classification. This is the model architecture that we will replicate:
<img src="./Images/7.png" title="" alt="" width="821">

Why do we need sigmoid function, and not just take such that input>0 = category 1 else category 0 as shown on the image below?
<img src="./Images/8.png" title="" alt="" width="821">
so it stops the values getting too large so that the loss functions do no deal with large number, with itself being too large

Jupyter notebook (continuation):

```python
# creating data
n_per_clust = 100
blur = 1

# so that data A is centered aroung (1,1), and data B centered around (5,1)
A = [1, 1]
B = [5, 1]

a = [A[0] + np.random.randn(n_per_clust) * blur, A[1] + np.random.randn(n_per_clust)]
b = [B[0] + np.random.randn(n_per_clust) * blur, B[1] + np.random.randn(n_per_clust)]

labels_np = np.vstack((np.zeros((n_per_clust,1)), np.ones((n_per_clust, 1))))

# merge a and b so they are not on different variables
data_np = np.hstack((a,b)).T

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()
# each item in labels as [0.] or [1.]
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1], 'bs')
ax.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1], 'ko')
ax.set(xlabel='dimension 1', ylabel='dimension 2')
plt.show()
```

<img src="./Images/9.png" title="" alt="" width="555">

```python
# (2,1) means it takes 2 inputs, and sends 1 output
ANNclassify = nn.Sequential(
    nn.Linear(2,1), # input layer
    nn.ReLU(),      # activation function
    nn.Linear(1,1), # output layer
    nn.Sigmoid()    # final activation function
)
ANNclassify
```

Sequential(
  (0): Linear(in_features=2, out_features=1, bias=True)
  (1): ReLU()
  (2): Linear(in_features=1, out_features=1, bias=True)
  (3): Sigmoid()
)

```python
learning_rate = 0.01
loss_func = nn.BCELoss()
# in "metaparameters", it is shown that BCEWithLogitsLoss is better that BCELoss
optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learning_rate)
```

```python
num_epochs = 1000
losses = torch.zeros(num_epochs)

for epochi in range(num_epochs):
    # forward pass
    y_hat = ANNclassify(data)
    # loss function
    loss = loss_func(y_hat, labels)
    losses[epochi] = loss
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```python
fig, ax = plt.subplots()
ax.plot(losses.detach(), 'o', markerfacecolor='w')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.show()
```

<img src="./Images/10.png" title="" alt="" width="481">

```python
# preds is a 2d tensor array of probabilities
# pred_labels just has True when preds item has prob>0.5, else False
preds = ANNclassify(data)
pred_labels = preds > 0.5 

misclassified = np.where(pred_labels != labels)[0] # indices of missclassified examples
total_acc = 100 - 100 * len(missclassified) / (2 * n_per_clust)
print(f'Total accuracy: {total_acc}')
```

Total accuracy: 50.0

```python
fig, ax = plt.subplots()
ax.plot(data[missclassified, 0], data[missclassified,1], 'rx', markersize=12, markeredgewidth=3)
ax.plot(data[np.where(~pred_labels)[0],0],data[np.where(~pred_labels)[0],1],'bs')
ax.plot(data[np.where(pred_labels)[0],0] ,data[np.where(pred_labels)[0],1] ,'ko')
plt.legend(['Missclassified', 'blue', 'black'], bbox_to_anchor=(1,1))
plt.show()
```

<img src="./Images/11.png" title="" alt="" width="613">

## Experiment on how learning rate affect model performance (Learning rates comparison: Video 48)

Jupyter notebook (continuation):

```python
# creating data
def create_data():
    n_per_clust = 100
    blur = 1

    # so that data A is centered aroung (1,1), and data B centered around (5,1)
    A = [1, 1]
    B = [5, 1]

    a = [A[0] + np.random.randn(n_per_clust) * blur, A[1] + np.random.randn(n_per_clust)]
    b = [B[0] + np.random.randn(n_per_clust) * blur, B[1] + np.random.randn(n_per_clust)]

    labels_np = np.vstack((np.zeros((n_per_clust,1)), np.ones((n_per_clust, 1))))

    # merge a and b so they are not on different variables
    data_np = np.hstack((a,b)).T

    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()
    # each item in labels as [0.] or [1.]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1], 'bs')
    ax.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1], 'ko')
    ax.set(xlabel='dimension 1', ylabel='dimension 2')
    plt.show()
    return data, labels
data, labels = create_data()
```

<img title="" src="./Images/12.png" alt="" width="452">

```python
def create_ann_model(learning_rate):
    ann_classify = nn.Sequential(
        nn.Linear(2,1),
        nn.ReLU(),
        nn.Linear(1,1),
#         nn.Sigmoid()
    )
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(ann_classify.parameters(), lr=learning_rate)
    return ann_classify, loss_func, optimizer
```

```python
num_epochs = 1000
def train_model(ann_classify, loss_func, optimizer, data, labels):
    # storing for plot
    losses = torch.zeros(num_epochs)

    for epochi in range(num_epochs):
        # forward prop
        y_hat = ann_classify(data)
        # compute loss
        loss = loss_func(y_hat, labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = ann_classify(data)
    total_acc = 100 * torch.mean(((preds>0) == labels).float())

    return losses, preds, total_acc
```

Test the functions above by running it once

```python
ann_classify, loss_func, optimizer = create_ann_model(0.01)
losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)
print(f'total accuracy {total_acc}')

fig, ax = plt.subplots()
ax.plot(losses.detach(), 'o', markerfacecolor='w')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.show()
```

total accuracy 95.5

<img src="./Images/13.png" title="" alt="" width="571">

```python
learning_rates = np.linspace(0.001, 0.1, 40)
acc_by_lr = []
all_losses = np.zeros((len(learning_rates), num_epochs)) # (40,1000) 2d array

for i, lr in enumerate(learning_rates):
    ann_classify, loss_func, optimizer = create_ann_model(lr)
    losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)

    # store results for plot
    acc_by_lr.append(total_acc)
    all_losses[i,:] = losses.detach()
```

We can observe from the first graph, either the model does very well or it does not do well at all, and this is because the times it did well, it got lucky with the random weight initializations.

This may due to the fact there are bunch of equally good local minimas, where the model performed well, but there are also equally bad local minimas, where the model got stuck and did poorly

```python
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,4))
ax0.plot(learning_rates, acc_by_lr, 's-')
ax0.set(xlabel='Learning rate', ylabel='Accuracy', title='Accuracy by learning rate')
ax1.plot(all_losses.T)
ax1.set(xlabel='Epoch number', ylabel='Loss', title='Loss by learning rate')
plt.show()
```

![](./Images/14.png)

## Improving previous section's model by adding more layers: (Multilayer ANN: Video 49)

The model on the previous section either did very well at 90% or performed by chance at 50%, in this section we explore a viable option, which is to add more layers. The next section explores how it can be improved actually

Jupyter notebook (continuation):

```python
data, labels = create_data() # from previous section
```

<img src="./Images/15.png" title="" alt="" width="376">

```python
def create_model(learning_rate):
    ann_classify = nn.Sequential(
        nn.Linear(2,16),   # input layer
        nn.ReLU(),         # activation function
        nn.Linear(16,1),   # hidden layer
        nn.ReLU(),         # activation function
        nn.Linear(1,1),    # output layer
        nn.Sigmoid()       # activation function
    )
    loss_func = nn.BCELoss() # but better to use BCEWithLogitsLoss
    optimizer = torch.optim.SGD(ann_classify.parameters(), lr=learning_rate)
    return ann_classify, loss_func, optimizer
```

```python
num_epochs = 1000
def train_model(ann_model, loss_func, optimizer, data, labels):
    losses = torch.zeros(num_epochs)
    for epochi in range(1000):
        # forward prop
        y_hat = ann_model(data)
        # loss function
        loss = loss_func(y_hat, labels)
        losses[epochi] = loss
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    preds = ann_model(data)
    # previous section it was 0 in place 0.5 because now we have sigmoid 
    # explicitly on the output layer, so 0.5 is the decision boundary
    total_acc = 100 * torch.mean(((preds > 0.5) == labels).float())

    return losses, preds, total_acc
```

Running it once to test the functions above...

```python
ann_classify, loss_func, optimizer = create_model(0.01)
losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)
print(f'Final accuracy {total_acc}%')
fig, ax = plt.subplots()
ax.plot(losses.detach(), markerfacecolor='w')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.show()
```

<img src="./Images/16.png" title="" alt="" width="525">

Experimenting again just like previous section...

```python
learning_rates = np.linspace(0.001, 0.1, 50)
acc_by_lr = []
all_losses = np.zeros((len(learning_rates), num_epochs))

for i, lr in enumerate(learning_rates):
    ann_classify, loss_func, optimizer = create_model(lr)
    losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)

    acc_by_lr.append(total_acc)
    all_losses[i,:] = losses.detach()
```

The conclusion is the same as before, so nothing has been improved, but this is a viable option

```python
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,4))
ax0.plot(learning_rates, acc_by_lr, 's-')
ax0.set(xlabel='Learning rate', ylabel='Accuracy', title='Accuracy by learning rate')
ax1.plot(all_losses.T)
ax1.set(xlabel='Epoch number', ylabel='Loss', title='Loss by learning rate')
plt.show()
```

<img src="./Images/17.png" title="" alt="" width="817">

## Improving previous section's model by adding more layers: (Multilayer ANN: Video 49)

The model on the previous section either did very well at 90% or performed by chance at 50%, in this section we explore a viable option, which is to add more layers. The next section explores how it can be improved actually

Jupyter notebook (continuation):

```python
data, labels = create_data() # from previous section
```

<img src="./Images/18.png" title="" alt="" width="451">

```python
def create_model(learning_rate):
    ann_classify = nn.Sequential(
        nn.Linear(2,16),   # input layer
#         nn.ReLU(),         # activation function
        nn.Linear(16,1),   # hidden layer
#         nn.ReLU(),         # activation function
        nn.Linear(1,1),    # output layer
        nn.Sigmoid()       # activation function
    )
    loss_func = nn.BCELoss() # but better to use BCEWithLogitsLoss
    optimizer = torch.optim.SGD(ann_classify.parameters(), lr=learning_rate)
    return ann_classify, loss_func, optimizer
```

```python
num_epochs = 1000
def train_model(ann_model, loss_func, optimizer, data, labels):
    losses = torch.zeros(num_epochs)
    for epochi in range(1000):
        # forward prop
        y_hat = ann_model(data)
        # loss function
        loss = loss_func(y_hat, labels)
        losses[epochi] = loss
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    preds = ann_model(data)
    # previous section it was 0 in place 0.5 because now we have sigmoid 
    # explicitly on the output layer, so 0.5 is the decision boundary
    total_acc = 100 * torch.mean(((preds > 0.5) == labels).float())

    return losses, preds, total_acc
```

```python
ann_classify, loss_func, optimizer = create_model(0.01)
losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)
print(f'Final accuracy {total_acc}%')
fig, ax = plt.subplots()
ax.plot(losses.detach(), markerfacecolor='w')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.show()
```

<img src="./Images/19.png" title="" alt="" width="423">

```python
learning_rates = np.linspace(0.001, 0.1, 50)
acc_by_lr = []
all_losses = np.zeros((len(learning_rates), num_epochs))

for i, lr in enumerate(learning_rates):
    ann_classify, loss_func, optimizer = create_model(lr)
    losses, preds, total_acc = train_model(ann_classify, loss_func, optimizer, data, labels)

    acc_by_lr.append(total_acc)
    all_losses[i,:] = losses.detach()
```

In conclusion, this actually improved the model as accuracy remained high (not being hit or miss) and that is because the dataset is linearly separable so a linear separator will outperform a non-linear separator.

In case of non-linear separator that got used in the previous 2 sections, the model was forced to search for more complex solutions to the problem, in which the solution is quite simple. Logistic Regression could also work better that non-linear deep learning model in this case.

```python
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,4))
ax0.plot(learning_rates, acc_by_lr, 's-')
ax0.set(xlabel='Learning rate', ylabel='Accuracy', title='Accuracy by learning rate')
ax1.plot(all_losses.T)
ax1.set(xlabel='Epoch number', ylabel='Loss', title='Loss by learning rate')
plt.show()
```

![](./Images/20.png)

## (Why multilayer linear models don't exist: Video 51)

The model on the previous section uses a linear model with multiple layers, but it is actually equivalent to having 1 layer, and not multiple layers, if there are no non-linearity in between, through activation functions

A multi-layer linear model is really just 1 layer, as long as there are no non-linear activation units

## (Multi-output ANN (iris dataset): Video 52)

In this section, we build a new model to recognize 3 different types of iris flowers: setosa, versicolor, verginica.
The model architecture is as follows:
<img src="./Images/21.png" title="" alt="" width="821">

## Multi class classification softmax (Multi-output ANN (iris dataset): Video 52)

In this section, we build a new model to recognize 3 different types of iris flowers: setosa, versicolor, verginica.
The model architecture is as follows:
<img src="./Images/21.png" title="" alt="" width="821">

Jupyter notebook (continuation):

```python
iris = sns.load_dataset('iris')
iris.head()
```

<img src="./Images/22.png" title="" alt="" width="496">

```python
data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == 'setosa'] = 1 # no need
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2
data, labels
```

<img src="./Images/23.png" title="" alt="" width="537">

```python
def create_model(lr):
    ann_iris = nn.Sequential(
        nn.Linear(4,64),  # input layer
        nn.ReLU(),        # activation function
        nn.Linear(64,64), # hidden layer
        nn.ReLU(),        # activation function
        nn.Linear(64,3)   # output layer
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ann_iris.parameters(), lr=lr)
    return ann_iris, loss_func, optimizer
```

```python
num_epochs = 1000
def train_model(model, loss_func, optimizer, data, labels):
    losses = torch.zeros(num_epochs)
    ongoing_acc = []
    y_hat = np.zeros((len(data), 3))

    for epochi in range(num_epochs):
        # forward prop
        y_hat = model(data)
        # loss func
        loss = loss_func(y_hat, labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute accuracy
        # assign the cat based on the col with largest value i.e
        # if col1 has largest value, then category 1
        matches = torch.argmax(y_hat, axis=1) == labels
        matches_numeric = matches.float()
        accuracy_perc = 100 * torch.mean(matches_numeric)
        ongoing_acc.append(accuracy_perc)

    preds = model(data)
    pred_labels = torch.argmax(preds, axis=1)
    total_acc = 100 * torch.mean((pred_labels == labels).float())
    return total_acc, losses, ongoing_acc, y_hat
```

```python
ann_iris, loss_func, optimizer = create_model(0.01)
total_acc, losses, ongoing_acc, y_hat = train_model(ann_iris, loss_func, optimizer, data, labels)
total_acc
```

tensor(98.)

```python
print(f'final accuracy {total_acc}%')
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(13,4))
ax0.plot(losses.detach())
ax0.set(xlabel='Epoch', ylabel='Loss', title='Losses')

ax1.plot(ongoing_acc)
ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
plt.show()
```

<img src="./Images/24.png" title="" alt="" width="850">

first 1/3 is setosa, next 1/3 is versicolor, and the last 1/3 is virginica. The graph talks about model's predicted outputs. We can see on the first 1/3, the model had no difficulties in predicting setosa, since probability of setosa is close to 1, and the other 2 really low. However on the next 2 1/3s, the model had a bit of difficulty predicting versicolor and virginica

```python
sm = nn.Softmax(1)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sm(y_hat.detach()), 's-', markerfacecolor='w')
ax.set(xlabel='Stimulus number', ylabel='Probability')
ax.legend(['setosa', 'versicolor', 'virginica'])
plt.show()
```

<img src="./Images/25.png" title="" alt="" width="859">

## Repeating the code on the previous section on different problem softmax multi class classification (CodeChallenge: more qwerties!: Video 53)

Jupyter notebook (continuation):

```python
# creating data
def create_data():
    n_per_clust = 100
    blur = 1

    # so that data A is centered aroung (1,1), and data B centered around (5,1)
    A = [1, 1]
    B = [5, 1]
    C = [3,-2]

    a = [A[0] + np.random.randn(n_per_clust) * blur, A[1] + np.random.randn(n_per_clust)*blur]
    b = [B[0] + np.random.randn(n_per_clust) * blur, B[1] + np.random.randn(n_per_clust)*blur]
    c = [C[0] + np.random.randn(n_per_clust) * blur, C[1] + np.random.randn(n_per_clust)*blur]

    labels_np = np.vstack((np.zeros((n_per_clust,1)),np.ones((n_per_clust,1)),1+np.ones((n_per_clust,1))))

    # merge a and b so they are not on different variables
    data_np = np.hstack((a,b,c)).T

    data = torch.tensor(data_np).float()
    labels = torch.squeeze(torch.tensor(labels_np).long())
    # each item in labels as [0.] or [1.]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1], 'bs')
    ax.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1], 'ko')
    ax.plot(data[np.where(labels==2)[0],0], data[np.where(labels==2)[0],1], 'r^')
    ax.set(xlabel='dimension 1', ylabel='dimension 2')
    plt.show()
    return data, labels
```

```python
def create_model(lr):
    ann_classify = nn.Sequential(
        nn.Linear(2,4),   # input layer
        nn.ReLU(),        # activation function
        nn.Linear(4,3),   # output layer
        nn.Softmax(dim=1) # activation function
    )
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ann_classify.parameters(), lr=lr)
    return ann_classify, loss_func, optimizer

num_epochs = 10000
def train_model(model, loss_func, optimizer, data, labels):
    losses = torch.zeros(num_epochs)
    ongoing_acc = []
    y_hat = np.zeros((len(data), 3))

    for epochi in range(num_epochs):
        # forward prop
        y_hat = model(data)
        # loss func
        loss = loss_func(y_hat, labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute accuracy
        # assign the cat based on the col with largest value i.e
        # if col1 has largest value, then category 1
        matches = torch.argmax(y_hat, axis=1) == labels
        matches_numeric = matches.float()
        accuracy_perc = 100 * torch.mean(matches_numeric)
        ongoing_acc.append(accuracy_perc)

    preds = model(data)
    pred_labels = torch.argmax(preds, axis=1)
    total_acc = 100 * torch.mean((pred_labels == labels).float())
    return total_acc, losses, ongoing_acc, y_hat
```

```python
data, labels = create_data()
ann_classify, loss_func, optimizer = create_model(0.01)
total_acc, losses, ongoing_acc, y_hat = train_model(ann_classify, loss_func, optimizer, data, labels)
total_acc
```

<img src="./Images/26.png" title="" alt="" width="465">

```python
print(f'final accuracy {total_acc}%')
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(13,4))
ax0.plot(losses.detach())
ax0.set(xlabel='Epoch', ylabel='Loss', title='Losses')

ax1.plot(ongoing_acc)
ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
plt.show()
```

![](./Images/27.png)

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_hat.detach(), 's-', markerfacecolor='w')
ax.set(xlabel='Stimulus number', ylabel='Probability')
ax.legend(['cat 1', 'cat 2', 'cat 3'])
plt.show()
```

![](./Images/28.png)

## Comparing number of neurons in a hidden layer affecting model performance (Comparing the number of hidden units: Video 54)

The model architecture is as follows. The middle layer has its neurons varied:
<img src="./Images/29.png" title="" alt="" width="821">

Jupyter notebook (continuation):

```python
iris = sns.load_dataset('iris')
iris.head()
```

<img title="" src="./Images/30.png" alt="" width="395">

```python
data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == 'setosa'] = 1 # no need
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2
data, labels
```

<img src="./Images/31.png" title="" alt="" width="456">

```python
def create_model(n_hidden):
    ann_iris = nn.Sequential(
        nn.Linear(4,n_hidden),  # input layer
        nn.ReLU(),        # activation function
        nn.Linear(n_hidden,n_hidden), # hidden layer
        nn.ReLU(),        # activation function
        nn.Linear(n_hidden,3)   # output layer
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ann_iris.parameters(), lr=0.01)
    return ann_iris, loss_func, optimizer

num_epochs = 150
def train_model(model, loss_func, optimizer, data, labels):
    y_hat = np.zeros((len(data), 3))

    for epochi in range(num_epochs):
        # forward prop
        y_hat = model(data)
        # loss func
        loss = loss_func(y_hat, labels)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute accuracy
        # assign the cat based on the col with largest value i.e
        # if col1 has largest value, then category 1
        matches = torch.argmax(y_hat, axis=1) == labels
        matches_numeric = matches.float()
        accuracy_perc = 100 * torch.mean(matches_numeric)

    preds = model(data)
    pred_labels = torch.argmax(preds, axis=1)
    total_acc = 100 * torch.mean((pred_labels == labels).float())
    return total_acc, y_hat
```

```python
num_hiddens = np.arange(1,129)
accuracies = []

for n_units in num_hiddens:
    ann_iris, loss_func, optimizer = create_model(n_units)
    total_acc, y_hat = train_model(ann_iris, loss_func, optimizer, data, labels)
    accuracies.append(total_acc)
```

```python
fig, ax = plt.subplots(1, figsize=(12,6))
ax.plot(accuracies,'ko-',markerfacecolor='w',markersize=9)
ax.plot(num_hiddens[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(num_hiddens[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.set(ylabel='accuracy', xlabel='Number of hidden units', title='Accuracy')
plt.show()
```

<img src="./Images/32.png" title="" alt="" width="761">

## Comparing if a wider, or a deeper model is better (Depth vs. breadth: number of parameters: Video 55)

On this section, breadth and depth are talked about and how they impact the accuracy of the model:
<img src="./Images/33.png" title="" alt="" width="761">

We compare these 2 models, depth vs width, as to which can fit more parameters:
<img src="./Images/34.png" title="" alt="" width="761">

The wide model has 27 parameters by 20 weights (number of arrows into each node) + 7 biases (each node has 1 bias) = 27

Jupyter notebook (continuation):

```python
wide_net = nn.Sequential(
    nn.Linear(2,4),  # hidden layer
    nn.Linear(4,3)   # output layer
)

deep_net = nn.Sequential(
    nn.Linear(2,2),  # hidden layer
    nn.Linear(2,2),  # hidden layer
    nn.Linear(2,3)   # output layer
)

wide_net, deep_net
```

(Sequential(
   (0): Linear(in_features=2, out_features=4, bias=True)
   (1): Linear(in_features=4, out_features=3, bias=True)
 ),
 Sequential(
   (0): Linear(in_features=2, out_features=2, bias=True)
   (1): Linear(in_features=2, out_features=2, bias=True)
   (2): Linear(in_features=2, out_features=3, bias=True)
 ))

2 bias in first layer because of 2 nodes and so on...

```python
# checking parameters
for p in deep_net.named_parameters():
    print(p)
    print('')
```

('0.weight', Parameter containing:
tensor([[-0.2266,  0.6159],
        [-0.5427,  0.3914]], requires_grad=True))
('0.bias', Parameter containing:
tensor([ 0.0601, -0.6176], requires_grad=True))
('1.weight', Parameter containing:
tensor([[-0.6173, -0.0172],
        [ 0.2947,  0.3037]], requires_grad=True))
('1.bias', Parameter containing:
tensor([-0.0281,  0.4915], requires_grad=True))
('2.weight', Parameter containing:
tensor([[ 0.0926,  0.2777],
        [-0.6431,  0.6447],
        [-0.0813,  0.2267]], requires_grad=True))
('2.bias', Parameter containing:
tensor([ 0.3679,  0.5911, -0.1489], requires_grad=True))

```python
# counting number of nodes
# instead counting number of biases since number of nodes = number of biases
num_nodes_wide = 0
for param_name,param_vect in deep_net.named_parameters():
    if 'bias' in param_name:
        num_nodes_wide += len(param_vect)
num_nodes_wide
```

7

.parameters() same as in .named_parameters(), pointing to the same component, just that .parameters(), it is does not get the initial part of the tuple containing the names

```python
for p in wide_net.parameters():
    print(p)
    print('')
```

Parameter containing:
tensor([[-0.4053, -0.2636],
        [ 0.4693,  0.5064],
        [-0.0145,  0.6735],
        [ 0.3260,  0.0499]], requires_grad=True)
Parameter containing:
tensor([ 0.2334, -0.5238,  0.5488,  0.3017], requires_grad=True)
Parameter containing:
tensor([[-0.3958, -0.2661, -0.3754,  0.1079],
        [-0.3857, -0.4818,  0.4816,  0.2060],
        [ 0.4674, -0.2042,  0.4493,  0.0324]], requires_grad=True)
Parameter containing:
tensor([0.3476, 0.2666, 0.0050], requires_grad=True)

```python
# counting number of trainable parameters
# testing requires_grad for true
# if requires_grad == false, then the parameters are fixed and not trainable
n_params = 0
for p in wide_net.parameters():
    if p.requires_grad:
        print(f'this piece has {p.numel()} parameters')
        n_params += p.numel()
print(f'total number of params in wide model: {n_params}')

# for loop above equivalent to...
n_params_deep = np.sum([p.numel() for p in deep_net.parameters() if p.requires_grad])
print(f'total number of params in deep model: {n_params_deep}')
```

this piece has 8 parameters
this piece has 4 parameters
this piece has 12 parameters
this piece has 3 parameters
total number of params in wide model: 27
total number of params in deep model: 21

```python
torchinfo.summary(wide_net, (1,2))
```

<img src="./Images/35.png" title="" alt="" width="825">

## Better way of doing nn.Sequential (Defining models using sequential vs. class: Video 56)

Both codes below do the same thing, and nn.Sequential is a shorter way of doing but it limits in some regards.

<img title="" src="./Images/36.png" alt="" width="559">

Jupyter notebook (continuation):

```python
# creating data
def create_data():
    n_per_clust = 100
    blur = 1

    # so that data A is centered aroung (1,1), and data B centered around (5,1)
    A = [1, 1]
    B = [5, 1]

    a = [A[0] + np.random.randn(n_per_clust) * blur, A[1] + np.random.randn(n_per_clust) * blur]
    b = [B[0] + np.random.randn(n_per_clust) * blur, B[1] + np.random.randn(n_per_clust) * blur]

    labels_np = np.vstack((np.zeros((n_per_clust,1)), np.ones((n_per_clust, 1))))

    # merge a and b so they are not on different variables
    data_np = np.hstack((a,b)).T

    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()
    # each item in labels as [0.] or [1.]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1], 'bs')
    ax.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1], 'ko')
    ax.set(xlabel='dimension 1', ylabel='dimension 2')
    plt.show()
    return data, labels
data, labels = create_data()
```

<img title="" src="./Images/37.png" alt="" width="235">

```python
class ann_model(nn.Module):
    def __init__ (self):
        super().__init__()
        # initilizing the layers
        self.input = nn.Linear(2,1)  # input layer
        self.output = nn.Linear(1,1) # output layer
    def forward(self,x):
        # initilizing operations on the layers
        # pass through input layers
        x = self.input(x)
        # apply relu
        x = F.relu(x)
        # output layer
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
```

```python
ann_classify = ann_model()

learning_rate = 0.01
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(ann_classify.parameters(), lr=learning_rate)
```

```python
num_epochs = 1000
losses = torch.zeros(num_epochs)

for epochi in range(num_epochs):
    y_hat = ann_classify(data) # forward prop
    
    loss = loss_func(y_hat, labels) # loss function
    losses[epochi] = loss
    optimizer.zero_grad()  # back prop
    loss.backward()
    optimizer.step()
```

```python
fig, ax = plt.subplots()
ax.plot(losses.detach(), 'o', markerfacecolor='w')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.show()
```

<img src="./Images/38.png" title="" alt="" width="429">

```python
preds = ann_classify(data)
pred_labels = preds > 0.5
misclassified = np.where(pred_labels != labels)[0]
total_acc = 100 - 100 * len(misclassified)/(2*n_per_clust)
print(f'total accuracy: {total_acc}')
```

total accuracy: 93.5

```python
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(data[misclassified,0] ,data[misclassified,1],'rx',markersize=12,markeredgewidth=3)
ax.plot(data[np.where(~pred_labels)[0],0],data[np.where(~pred_labels)[0],1],'bs')
ax.plot(data[np.where(pred_labels)[0],0] ,data[np.where(pred_labels)[0],1] ,'ko')
ax.legend(['Misclassified','blue','black'],bbox_to_anchor=(1,1))
ax.set_title(f'{total_acc}% correct')
plt.show()
```

<img src="./Images/39.png" title="" alt="" width="376">x



## Experiment varying both hidden layers and number of neurons per hidden layer to see if depth or width of model matters more(Model depth vs. breadth: Video 57)

Jupyter notebook (continuation):

```python
iris = sns.load_dataset('iris')
# convert from pandas dataframe to tensor
data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == 'setosa'] = 0 # no need
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2
```

```python
class ann_iris(nn.Module):
    def __init__(self, n_neurons, n_layers):
        # initializing the layers
        super().__init__()
        
        # create dictionary to store the layers
        self.layers = nn.ModuleDict()
        self.n_layers = n_layers
        
        # input layer, 4 implying number of input features
        self.layers['input'] = nn.Linear(4, n_neurons)
        
        # hidden layer
        for i in range(n_layers):
            self.layers[f'hidden{i}'] = nn.Linear(n_neurons, n_neurons)
        
        # output layer
        self.layers['output'] = nn.Linear(n_neurons, 3)
    
    # forward prop
    def forward(self, x):
        # initializing the operations on the layers
        x = self.layers['input'](x)
        
        # hidden layers
        for i in range(self.n_layers):
            x = F.relu(self.layers[f'hidden{i}'](x))
        
        x = self.layers['output'](x)
        return x      
```

```python
# testing the code of the class
n_units_per_layer = 12
n_layers = 4
net = ann_iris(n_units_per_layer, n_layers)

tmpx = torch.randn(10,4)
y = net(tmpx)
print(y.shape)
print(y)
```

torch.Size([10, 3])
tensor([[-0.2527, -0.1492,  0.0902],
        [-0.2515, -0.1470,  0.0954],
        [-0.2511, -0.1515,  0.0928],
        [-0.2518, -0.1526,  0.0941],
        [-0.2497, -0.1451,  0.0918],
        [-0.2495, -0.1474,  0.0969],
        [-0.2556, -0.1456,  0.0807],
        [-0.2515, -0.1516,  0.0872],
        [-0.2489, -0.1456,  0.0946],
        [-0.2517, -0.1443,  0.0872]], grad_fn=<AddmmBackward0>)

```python
def train_model(model, data, labels):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epochi in range(num_epochs):
        y_hat = model(data) # forward prop
        loss = loss_func(y_hat, labels) # loss func
        optimizer.zero_grad() # back prop
        loss.backward()
        optimizer.step()
        
        preds = model(data)
        pred_labels = torch.argmax(preds, axis=1)
        acc = 100 * torch.mean((pred_labels == labels).float())
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return acc, n_params
```

```python
num_layers = range(1,6) # number of hidden layers
num_neurons = np.arange(4,101,3) # number of neurons per hidden layer

accuracies = np.zeros((len(num_neurons), len(num_layers)))
total_params = np.zeros((len(num_neurons), len(num_layers)))

num_epochs = 500

for neuron_idx in range(len(num_neurons)):
    for layer_idx in range(len(num_layers)):
        net = ann_iris(num_neurons[neuron_idx], num_layers[layer_idx])
        acc, n_params = train_model(net, data, labels)
        
        accuracies[neuron_idx, layer_idx] = acc
        total_params[neuron_idx, layer_idx] = n_params
```

```python
fig, ax = plt.subplots(1, figsize=(12,6))
ax.plot(num_neurons,accuracies,'o-',markerfacecolor='w',markersize=9)
ax.plot(num_neurons[[0,-1]], [33,33], '--', color=[0.8,0.8,0.8])
ax.plot(num_neurons[[0,-1]],[67,67],'--', color=[0.8,0.8,0.8])
ax.set(xlabel='Number of hidden neurons', ylabel='Accuracy', title='Accuracy')
plt.legend(num_layers)
plt.show()
```

<img src="./Images/40.png" title="" alt="" width="689">

0 Correlation means no relation between accuracy and number of parameters

```python
x = total_params.flatten()
y = accuracies.flatten()

r = np.corrcoef(x,y)[0,1]
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x,y,'o')
ax.set(xlabel='number of parameters', ylabel='accuracy', title=f'correlation: r={str(np.round(r,3))}')
plt.show()
```

<img src="./Images/41.png" title="" alt="" width="366">

In conclusion:

* Deeper models are not necessarily better. They require more training and FLOPS

* Model performance is not simply a function of number of trainable parameters. Architecture matters.

* Shallow models learn fast, but deeper models can learn more complex mappings.

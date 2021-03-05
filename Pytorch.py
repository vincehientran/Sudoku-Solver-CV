import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import Model
import numpy as np

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# peprare data set
trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST(root = './data', train = False, download =False, transform = transforms.Compose([transforms.ToTensor()]))
valset , testset0 = torch.utils.data.random_split(testset,[int(0.9*len(testset)), int(0.1*len(testset))])

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(testset0, batch_size=1, shuffle=False)

# visualize data
fig = plt.figure(figsize=(20,10))
for i  in range(1,6):
    img =  transforms.ToPILImage(mode='L')(trainset[i][0])
    fig.add_subplot(1,6,i)
    plt.title(trainset[i][1])
    plt.imshow(img)
plt.show()

model = Model()
criterion = torch.nn.CrossEntropyLoss()
# parameters = list(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if(torch.cuda.is_available()):
    model.cuda()

no_epochs = 10
train_loss = list()
val_loss = list()
best_val_loss = 1
for epoch in range(no_epochs):
    print(epoch)
    print("iteration")
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    #training
    for itr, (image, label) in enumerate(train_dataloader):

        if(torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        optimizer.zero_grad()             # zero out the optimizer gradients

        pred = model(image)               # prediction for batch

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()                   # backpropagete the loss
        optimizer.step()                  # modify the model params

    total_train_loss = total_train_loss / (itr + 1)
    train_loss.append(total_train_loss)

    # validation
    model.eval()                         # set the midel to eval mode
    totoal = 0
    for itr, (image, label) in enumerate(val_dataloader):

        if(torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                totoal = totoal + 1

    accuracy = totoal / len(valset)

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)

    print('\nEpoch:{}/{}, Train loss: {:.8f}, val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs,total_train_loss,total_val_loss, accuracy))

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        print('save')
        torch.save(model.state_dict(), "model.dth")

fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()

# test model
model.load_state_dict(torch.load("model.dth"))           # fetch the saved model stae dic having the lowest validation loss
model.eval()                                             # model eval mode

results = list()
total = 0
for itr, (image, label) in enumerate(test_dataloader):

        if(torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)
        pred = torch.nn.functional.softmax(pred, dim=1)

        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
                results.append((image, torch.max(p.data, 0)[1]))    # result is a  list of tuple of image and predicted label
                print(label[i])
                print(torch.max(p.data, 0)[1])
                print(image.size())
test_accuracy = total / (itr + 1)
print('Test accuracy: {:.8f}'.format(test_accuracy))

# visualize results
fig=plt.figure(figsize=(20, 10))
for i in range(1, 11):
    img = transforms.ToPILImage(mode='L')(results[i][0][0].squeeze(0).detach().cpu())
    fig.add_subplot(2, 5, i)
    plt.title(results[i][1].item())
    plt.imshow(img)
plt.show()

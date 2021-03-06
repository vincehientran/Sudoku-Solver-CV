import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from Model import Model
import numpy as np

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# peprare data set
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST(root = './data', train = False, download =True, transform = transforms.Compose([transforms.ToTensor()]))
valset , testset0 = torch.utils.data.random_split(testset,[int(0.9*len(testset)), int(0.1*len(testset))])

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(testset0, batch_size=1, shuffle=False)

model = Model()
# test model
model.load_state_dict(torch.load("model.dth"))           # fetch the saved model stae dic having the lowest validation loss
model.eval()                                             # model eval mode

results = list()
total = 0
test = None
for itr, (image, label) in enumerate(test_dataloader):

        if(torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        test = image
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

print(test)

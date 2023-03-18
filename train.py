import torch
import torch.nn as nn
from tqdm import tqdm

from model import SimpleCNN
from data import get_data


def get_accuray(preds, labels):
    count = 0
    for i in range(0,len(labels)):
        if labels[i] == preds[i]:
            count += 1
        else:
            pass
    return count/len(preds)


trainloader, testloader, train, test = get_data()


# logging the training configrations to weights and biases
config = {
  "learning_rate": 0.001,
  "momentum" : 0.9,
  "epochs": 15,
  "batch_size": 100
}


net = SimpleCNN() # Load model
loss_fn = nn.CrossEntropyLoss() # define loss functions
optimizer = torch.optim.SGD(net.parameters(), lr= config['learning_rate'], momentum= config['momentum']) # setting hyperparameters


Loss, Acc, best_acc = [], [], 0.7

for i in range(0,80):
    loss_, acc_,  count = 0, 0, 0
    for img, label in tqdm(trainloader):
        x = img
        y = label

        optimizer.zero_grad()
        net.zero_grad()

        output = net(x)
        loss = loss_fn(output, y)
        
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)
        acc = get_accuray(preds, label)

        loss_ = loss_ + loss
        acc_ = acc_ + acc
        count += 1

    # wandb.log({"loss": loss_/count, 
    #            "Acc": acc_/count })
    print({"loss": loss_/count, 
               "Acc": acc_/count })

    if acc_/count > best_acc:
        # wandb.run.summary["best_accuracy"] = acc_/count
        torch.save(net.state_dict(), "model.pts")
        best_accuracy = acc_
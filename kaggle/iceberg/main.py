import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_model_summary
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import *
from model import CNN


# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device type is', device)


# hyperparameter
PATH = './weights/'
num_epochs = 50
batch_size = 32
learning_rate = 1e-2

def training(train_loader, valid_loader, model, criterion, optimizer):
    # train
    model.train()
    train_accuracy = []
    valid_accuracy = []
    print('Training Start')
    for epoch in range(num_epochs):
        losses = []
        corrects = 0
        total = 0

        for i, (ice_imgs, labels) in enumerate(train_loader):
            # GPU
            ice_imgs, labels = ice_imgs.to(device), labels.to(device)

            # forward
            scores = model(ice_imgs)
            loss = criterion(scores, labels)
            losses.append(loss.item()) # running loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running Accuracy
            _, predictions = scores.max(1)
            correct = (predictions == labels).sum()
            corrects += correct.item()
            total += ice_imgs.shape[0]


        train_acc = corrects/total*100
        valid_acc = valid(valid_loader, model)
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc.item())
        print('Epoch:[{}/{}] \tLoss:{:.6f}\tTrain acc:{:.2f}%\tValid acc:{:.2f}%'.format(
            epoch+1,num_epochs,sum(losses)/len(losses),train_acc,valid_acc)  )
        if (epoch+1) % 5 == 0:
            torch.save({
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch+1,
                'loss': sum(losses)/len(losses),
                'accuracy':corrects/total*100
            }, f'{PATH}/CNN_{epoch+1}.pth')
    
    plt.figure(figsize=(12,8))
    plt.plot(train_accuracy, color = 'blue', label='TRAIN')
    plt.plot(valid_accuracy, color = 'red', label='VALID')
    plt.legend()
    plt.title('CNN Training')
    plt.savefig('CNN.png')
    plt.close()



def valid(valid_loader, model):
    num_corrects = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_corrects += (predictions==y).sum()
            num_samples += x.shape[0]
    return num_corrects/num_samples*100


def getSubmission(test):
    # Dataset
    X_test, _ = getData(test)
    X_test = torch.from_numpy(X_test).float()
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)
    checkpoints = torch.load(f'{PATH}/CNN_50.pth')
    model.load_state_dict(checkpoints['model'])

    scores = []
    print('\n\nTest Processing!')
    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device)

            score = model(x)
            scores.extend(score.detach().cpu().numpy())
            
    scores = [nn.Sigmoid()(torch.from_numpy(x)) for x in scores]
    scores = [x.max() for x in scores]
    scores = [x.item() for x in scores]
    
    submission = pd.DataFrame()
    submission['id']=test['id']
    submission['is_iceberg']=scores
    submission.to_csv('sub.csv', index=False)

if __name__ == '__main__':
    
    # Load the data
    train = pd.read_json('../input/iceberg/train.json')
    test = pd.read_json('../input/iceberg/test.json')
    
    # Image Data and label
    X_train, y_train = getData(train, 'train')
    
    
    # train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
    
    # tensor dataset
    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    y_train = torch.from_numpy(y_train)
    y_valid = torch.from_numpy(y_valid)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    # loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=401, shuffle=False)
    
    # model & loss & optimize
    model = CNN().to(device)
    print(pytorch_model_summary.summary(model, torch.zeros(1, 3, 75, 75).to(device), show_input=True))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # train & valid
    training(train_loader, valid_loader, model, criterion, optimizer)
    
    # test
    getSubmission(test)
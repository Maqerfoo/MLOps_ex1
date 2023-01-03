import argparse
import sys
import matplotlib.pyplot as plt

import torch
import click
from torch import optim
from torchvision import datasets, transforms
from torch import nn

from data import mnist
from model import MyAwesomeModel



@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=5, help="number of epochs to train for")
def train(lr,epochs):
    print("Training day and night")
    print(lr)
    print(epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for e in range(epochs):
        print("Epoch: {}/{}".format(e+1,epochs))
        running_loss = 0
        for images,labels in train_set:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        else:
            print(f"Training loss: {running_loss/len(train_set)}")
            losses.append(running_loss)
    plt.plot(list(range(epochs)),losses)
    plt.show()
    torch.save(model.state_dict(), 'trained_model.pth')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    criterion = nn.CrossEntropyLoss()

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    running_loss = 0
    for images,labels in test_set:
        images = images.view(images.shape[0],-1)
        output = model(images)
        loss = criterion(output,labels)
        running_loss += loss.item()
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))        
    print(f'Accuracy: {accuracy.item()*100}% \nLoss: {running_loss}')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    
import torch
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from tqdm import tqdm



class data():
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.MNIST_data = torchvision.datasets.MNIST(".",train=True,transform=transform, download=True)

        self.numbers = {"0":[],"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]}
        
        for num in tqdm(self.MNIST_data,ascii=True,desc="data processing"):
            self.numbers[str(num[1])].append(num)

    def online_sample(self,num):
        return self.numbers[str(num)][np.random.randint(len(self.numbers))]

class mlp(nn.Module):
    def __init__(self,input_size=28*28,output=1,hidden_size=28*14):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output))

    def forward(self,x):
        return self.net(torch.flatten(x))

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = data()
    network_mlp = mlp()
    optimizer = optim.Adam(network_mlp.parameters(),lr=0.001)

    EPOCHS = 80 
    tasks = 10
    number_of_tests = 20


    fig,axis = plt.subplots(2,5)
    fig2,axis2 = plt.subplots(2,5)
    axis2 = axis2.flatten()
    axis = axis.flatten()

    for task_num in tqdm(range(tasks),desc="tasks",unit="tasks",ascii=True):
        
        x_plot = [[] for o in range(task_num+1)]
        y_plot = [[] for o in range(task_num+1)]
        x_plot2 = [[] for o in range(task_num+1)]
        y_plot2 = [[] for o in range(task_num+1)]

        for i in range(EPOCHS):
            optimizer.zero_grad()
            input,label = dataset.online_sample(task_num)
            output = network_mlp(input)
            loss = nn.MSELoss()(output,torch.tensor([label],dtype=torch.float))
            loss.backward()
            optimizer.step()
            


            for j in range(task_num+1):
                input,label = dataset.online_sample(j)
                output = network_mlp(input)
                loss = nn.MSELoss()(output,torch.tensor([label],dtype=torch.float))

                x_plot2[j].append(i)
                y_plot2[j].append(loss.detach().item())
            
            for j in range(task_num+1):
                
                hits_and_missed = []
                for _ in range(number_of_tests):
                    input,label = dataset.online_sample(j)
                    output = network_mlp(input)
                    loss = nn.MSELoss()(output,torch.tensor([label],dtype=torch.float))
                    if(np.around(output.detach().item()) == label):
                        hits_and_missed.append(1)
                    else:
                        hits_and_missed.append(0)

                x_plot[j].append(i)
                accuracy = (np.sum(hits_and_missed)/len(hits_and_missed))*100
                y_plot[j].append(accuracy)
            
        for index in range(task_num+1):
            axis[task_num].plot(x_plot[index],y_plot[index],label=str(index))
            axis2[task_num].plot(x_plot2[index],y_plot2[index],label=str(index))
        
        axis2[task_num].legend()
        axis[task_num].legend()
        axis2[task_num].set_ylabel("loss")
        axis2[task_num].set_xlabel("steps current task")
        axis2[task_num].set_title("task "+str(task_num))
        axis[task_num].set_ylabel("accuracy")
        axis[task_num].set_xlabel("steps current task")
        axis[task_num].set_title("task "+str(task_num))
    
    fig.tight_layout(pad=0)
    fig2.tight_layout(pad=0)
    plt.show()

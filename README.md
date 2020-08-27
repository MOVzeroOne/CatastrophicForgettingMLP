# MNIST handwritten digits learned sequentially using MLP

After a task has been learned for n amount of steps, we switch to another task to learn never to repeat the same task.
In the plot one can see the performance (in terms of MSE loss) of the previous tasks during the training of the current.
The performance on older tasks decreases because of catastrophic forgetting.

Small note:
The task is defined as a regression task where the input is a flattend image and the output is the corresponding number. </br>
Different scale is used for each subplot.</br>
<br/>
<h2>Loss different tasks<h2/>
<img src="https://github.com/MOVzeroOne/CatastrophicForgettingMLP/blob/master/plot.PNG">
<br/>
<h2>Accuracy different tasks<h2/>
<img src="https://github.com/MOVzeroOne/CatastrophicForgettingMLP/blob/master/plot1.PNG">

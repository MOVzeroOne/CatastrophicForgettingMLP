# MNIST handwritten digits learned sequentially using MLP

After a task has been learned for n amount of steps, we switch to another task to learn never to repeat the same task.
In the plot one can see the performance (in terms of MSE loss) of the previous tasks during the training of the current.
The performance on older tasks decreases because of catastrophic forgetting.

# Zap Q-learning with Nonlinear Function Approximation

This code is an implementation of our paper: "[Zap Q-learning with Nonlinear Function Approximation](https://arxiv.org/abs/1910.05405). S. Chen, A. Devraj, F. Lu, A. Busic and S. P. Meyn." 

It uses neural network to approximate the optimal Q-function and applies our Zap Q-learning algorithm to solve the Cartpole problem. Adaptations of the code to solve other examples in OpenAI gym are straightforward.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Our code is based on Python 2.7, Pytorch 1.4, numpy, OpenAI gym and etc.

## Training

To train the model(s) in the paper, run this command:

```train
python zapNN.py
```

> 📋 You can modifiy the network structure inside the code. It also provides two types of step-size schedules: decreasing step-sizes and constant step-sizes. Detailed definitions can be found in the paper.

## Evaluation

To reproduce the plots in our paper, simply run the following command following Training:

```eval
python eval_plot.py
```


## Contributing

> 📋 All content is licensed under the MIT license.
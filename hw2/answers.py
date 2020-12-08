r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
The Jacobian tensor would contain the derivatives of every output w.r.t. every 
input, for every sample in the batch, thus the resulting Jacobian tensor would
have size (N,Din,Dout) = (128,1024,2048)  

The above Jacobian tensor would contain 128*1024*2048 numbers, each of which
requires 32 bits, for a total of 8,589,934,592 bits = 1 gigabyte 
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # DONE: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 3
    lr = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # DONE: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.001, 0.01, 0.01, 0.02, 0.001
    #wstd = 10
    #lr_vanilla = 0.1
    #lr_momentum = 0.01
    #lr_rmsprop = 0.1
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


The graphs describe a trend in which models with greater dropout performed worse on the training set metrics
but better on the test metrics, with the difference growing with the number of iterations.  For example, at
iteration 1200 it can be clearly seen that the model without dropout has the best loss and accuracy on the
training set (roughly 0.3 cross-entropy loss and over 85% accuracy vs. roughly 1.2 cross-entropy loss and
25% accuracy for the model with the highest dropout probability), while the dropout models perform far
better than the model without dropout at the same iteration on the test data, both with a mean of roughly
2 cross-entropy loss and 25-27% accuracy (27% for the low-dropout model and 25% for the high) vs. cross-entropy loss 3 and 22% mean accuracy for the model
without dropout).  This matches what I would expect to see, since dropout is a form of regularization
(it may be thought of as "forcing the model not to rely too much on a single hidden feature of the input
but to rely instead on a combination of indicating factors in the input's hidden features", by causing models
that rely too much on a small number of hidden features to perform poorly when those features are dropped).
A desirable regularization (as defined in class) is one that reduces test error at the cost of increasing 
training error, by preventing overfitting to the particular input data given.  The graphs also show various 
notable differences between the results of the low-dropout setting and the high-dropout setting, namely that 
the high dropout setting seems to behave more poorly on the training data on both metrics (which is to be expected, 
given more regularization), but it is noteworthy that although both settings seem to achieve similar cross-entropy 
losses on the testing data, it was actually the low-dropout setting that led to the highest accuracy.  Although
this trend may be disproven given more iterations, it appears that the low-dropout setting is more suitable for
achieving high accuracy on the given data, likely due to overregulation when the dropout probability is overly
high at 0.8, causing the model to lack the necessary capacity to optimally classify the inputs.  Referring back 
to my previous description of the mechanism of dropout regularization, this may be thought of as "forcing the 
model to rely more on hidden features that are poor indicators of the class instead of focusing on the best 
indicators, in order to avoid poor performance when the best indicators are dropped".   


"""

part2_q2 = r"""
**Your answer:**


When training a model with cross-entropy loss, it is possible for the test loss to increase while
the accuracy also increases, due to the test loss' dependence on the (continuous) classification 
probabilities (a.k.a. "class scores") output by the model while the accuracy is dependent only on 
the final classification, which is taken to be the classification with the highest probability.  Thus, 
as long as the classification with the highest probability on some input remains the same, the
probabilities of the remaining classifications may be raised (thus harming/raising the test loss)
without affecting the test accuracy.  This may most easily be seen from the alternative equivalent 
definition for cross entropy loss derived in part 1:
$\begin{align} - x_y + \log\left(\sum_k e^{x_k}\right) \end{align}$
where it can be clearly seen that raising the probabilities ( $x_k$ ) of the remaining classifications 
increases the right term while not affecting the left turn.
This process of raising the likelihoods of the incorrect classifications without exceeding the maximal
likelihood while also selecting 1 additional incorrectly classified input and raising the correct 
correct classification until it becomes maximal would thus increase the accuracy (due to the single
additional correctly classified input while not changing the final classifications of any other input)
while also increasing the cross entropy loss (due to increasing the probabilities of the incorrect 
classifications for all the remaining inputs).  This may occur despite the model's being trained 
with the cross-entropy loss function either in an attempt to minimize the regularization term or
due to well-crafted input designed to abuse the stochastic descent mechanism of the optimization, by
ordering the inputs in an order such that the update step to minimize the loss function on one batch 
increases the value of the loss function on the next batch (this should be prevented by randomly 
selecting the batches out of the input database to make malicious ordering of the inputs impossible,
however an unlucky draw of batches could achieve the same effect). 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q6 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

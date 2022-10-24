#!/usr/bin/env python
# coding: utf-8

# ## Before you start working on the exercise
# 
# - Use Python version 3.7 up to 3.9. Make sure not to use Python 3.10
# - It is highly recommended to create a virtual environment for this course. You can find resources on how to create a virtual environment on the ISIS page of the course.
# - Make sure that no assertions fail or exceptions occur, otherwise points will be subtracted.
# - Use all the variables given to a function unless explicitly stated otherwise. If you are not using a variable you are doing something wrong.
# - Read the **whole** task description before starting with your solution.
# - After you submit the notebook more tests will be run on your code. The fact that no assertions fail on your computer locally does not guarantee that you completed the exercise correctly.
# - Please submit only the notebook file with its original name. If you do not submit an `ipynb` file you will fail the exercise.
# - Edit only between YOUR CODE HERE and END YOUR CODE.
# - Verify that no syntax errors are present in the file.
# - Before uploading your submission, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel\Restart) and then run all cells (in the menubar, select Cell\Run All).

# In[1]:


import sys

if (3,7) <= sys.version_info[:2] <= (3, 9):
    print("Correct Python version")
else:
    print(f"You are using a wrong version of Python: {'.'.join(map(str,sys.version_info[:3]))}")


# ![](https://www.tu-berlin.de/fileadmin/a70100710_summeruniversity/Summer_Uni_allgemein/Summer_Uni_Logo.png)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


import minified
from minified import max_allowed_loops

from unittest import TestCase
t = TestCase()


# In[ ]:





# In this exam you want to write an algorithm that finds a function $f(x)$ that closely represents two dimensional data.
# 
# * You do not need to understand how this algorithm works to complete these tasks. It is sufficient to read the bullet points to know what you _have to_ implement. The longer task descriptions explain what is happening.
# 
# * There are no loops allowed in _any_ of the excercises.
# 
# # $\textbf{Exercise 1: } \text{Creating the dataset (44 Points})$
#  
# In the first excercise you want to generate two dimensional data on which you can train and test your algorithm. You will first generate the x- and y-coordinates, split your data into train and test sets, and plot both of them.
# 
# <hr>
# 
# ### Exercise 1.1: Generate x-coordinates of the datapoints ( 6 points ) 
# 
# <hr>
# 
# First you will generate the x-coordinates using a uniform distribution. Since the y-coordinates will follow a specific pattern, their order will be relevant. Therefore you will need to return the x-coordinates in ascending order.
# 
# * Draw $N=500$ samples from the uniform distribution in the range $[ -4\pi, 2\pi )$.
# 
# $$\Large{
# x \sim \mathcal{U}niform[-4\pi, 2\pi),\quad x \in \mathbb{R}^{N}
# }$$
# 
# * Return $x$ sorted is ascending order.
# 
# 
# * Loops allowed in this excercise: 0

# In[3]:


@max_allowed_loops(0)
def generate_x(n_samples: int) -> np.ndarray:
    '''
    Create a np.ndarray vector containing `n_samples` sorted values drawn 
    from a uniform distribution in [-4π, 2π).
    '''
    # YOUR CODE HERE
    x = np.random.uniform(-4*np.pi, 2*np.pi, n_samples)
    x = np.sort(x)
    # YOUR CODE HERE
    
    return x


# In[4]:


n_samples = 500

x = generate_x(n_samples)
print(x[[0,1,2,3,4,-5,-4,-3,-2,-1]])
# running function again should yield different results
x2_size = n_samples + 10
x2 = generate_x(x2_size)

def check_x(x, size):
    assert x.shape == (size,)
    t.assertTrue(np.all(x >= - 4 * np.pi))
    t.assertTrue(np.all(x <= 2 * np.pi))
    np.testing.assert_array_equal(np.diff(x) >= 0, True, 'output is not sorted')
    


assert (x != x2[:n_samples]).mean() > 0.95
check_x(x, n_samples)
check_x(x2, x2_size)


# ### Exercise 1.2: Generate y-values of the datapoints ( 10 points ) 
# 
# <hr>
# 
# Since you want to simulate the behavior of the data that has a true function $f(x)$ to represent it, you need to define that function $f(x)$. Even though you will later want your algorithm to find a polynomial function to represent the data, the function you will use to generate it will not be polynomial. That means that there is not one perfect function the algorithm can find, as there would also not be in most real scenarios. Instead you will use sine and cosine functions to generate your data. Since in most real data there is noise, you will need to add normal (Gaussian) noise $\mathcal{E}$ to your data.
# 
# In this excercise you want to calculate a y-coordinate for each x-coordinate of your toy dataset.
# 
# * Define the actual function $f(x)$ to generate the y-coordinates: 
# 
# $$\Large{f(x) = \sin(x) + \cos(\frac x 2)}$$
# 
# * Generate some normal (Gaussian) noise $\mathcal{E}$:
# 
# 
# $$\Large{
# \quad \mathcal{E} \sim \mathcal{N}(0, \sigma=0.5),\quad \mathcal{E} \in \mathbb{R}^{N}
# }$$
# 
# The noise can be generated using a numpy function. The standard deviation of the noise should be 0.5.
# 
# * Calculate the y-coordinates by adding the normal Gaussian noise to $f(x)$:
# 
# $$\Large{y = f(x) + \mathcal{E}}$$
# 
# 
# * Loops allowed in this excercise: 0

# In[5]:


from typing import Tuple, Callable, TypeVar

@max_allowed_loops(0)
def generate_y(
    x: np.ndarray, std: float = 0.5
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Create an output for each sample created in the previous exercise.
    The underlying signal is `sin(x) + cos(x / 2)`. A normal noise with
    a standard deviation as provided in the argument `std` (the mean of the
    noise is 0) is added to the signal. The output of the function is a
    callable (a function) that outputs "clean" values (without noise) from
    the underlying signal, and the calculated noisy values calculated for 
    each sample in `x`.
    """
    # YOUR CODE HERE
    #noise
    noise = np.random.normal(0, std, len(x))
    f = lambda x: np.sin(x) + np.cos(x/2)
    y = f(x) + noise
    # YOUR CODE HERE
    
    return f, y


# In[6]:


f, y = generate_y(x, std=0.5)

assert callable(f), "f should behave like a function."
x_test = np.arange(3) * np.pi
np.testing.assert_allclose(f(x_test), [1, 0, -1], atol=1e-5)


def test_y(x, f, y, std):
    assert y.shape == x.shape
    clean_signal_difference = f(x) - y
    np.testing.assert_allclose(np.std(clean_signal_difference), std, atol=1e-1)


test_y(x, f, y, 0.5)

x2 = np.arange(10000)
std2 = 10
f2, y2 = generate_y(x2, std=std2)
test_y(x2, f2, y2, std2)


# ### Exercise 1.3: Split the data into training and test sets ( 10 points ) 
# 
# <hr>
# 
# You will now need to split up the data into a training and a test set. 
# 
# * The train set should contain 80% by default of the data while the test set contains the remaining datapoints.
# 
# * Both the train and the test values have to be sorted, in the same way the given dataset is sorted (ascending x-values).
# 
# * Return four numpy arrays: The x values of the training set, the y values of the training set, the x values of the test set and the y values of the test set.
# 
# 
# * Loops allowed in this excercise: 0

# In[7]:


from typing import Tuple
from sklearn.model_selection import train_test_split

@max_allowed_loops(0)
def split_data_and_sort(
    x: np.ndarray, y: np.ndarray, sp_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    '''
    Given x & y arrays and a split ratio, split the data in x & y
    into training and test arrays, such that the ratio of the length of the
    train arrays and the original array is as close to `sp_ratio` as possible.
    
    The returned train and test arrays are sorted again such that the x arrays
    are in ascending order, and the association between x and y is kept.
    '''
    # YOUR CODE HERE
    
    #split
    
    N = len(y)
    sp_idx = int(sp_ratio * N)
    
    idx = np.arange(N)
    np.random.shuffle(idx)
    
    tr_idx = idx[:sp_idx]
    te_idx = idx[sp_idx:]
    
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    
    x_te = x[te_idx]
    y_te = y[te_idx]
    
    #x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size = sp_ratio, shuffle=True)
    
    
    idx_sort_tr = np.argsort(x_tr)
    print(idx_sort_tr.shape)
    x_tr = x_tr[idx_sort_tr]
    y_tr = y_tr[idx_sort_tr]
    
    idx_sort_te = np.argsort(x_te)
    print(idx_sort_te.shape)
    x_te = x_te[idx_sort_te]
    y_te = y_te[idx_sort_te]
    #
    #argsort
    
    print(x_te)

    return x_tr, y_tr, x_te, y_te


# In[8]:


sp_ratio = 0.8
x_tr, y_tr, x_te, y_te = split_data_and_sort(x, y, sp_ratio)

assert np.all(np.diff(x_tr) >= 0), 'x_tr is not sorted'
assert np.all(np.diff(x_te) >= 0), 'x_te is not sorted'

assert len(x_tr) + len(x_te) == len(x)
assert len(y_tr) + len(y_te) == len(y)

assert len(x_tr) == int(sp_ratio * len(x))
assert len(y_tr) == int(sp_ratio * len(y))

assert len(x_te) == int((1 - sp_ratio) * len(x)) + 1
assert len(y_te) == int((1 - sp_ratio) * len(y)) + 1


x_tr2, *_ = split_data_and_sort(x, y, sp_ratio)
assert np.any(x_tr2 != x_tr)

x_tr3, _, x_te3, _ = split_data_and_sort(x, y, 0.5)
assert len(x_tr3) == len(x_te3)


# ### Exercise 1.4: Plot the data ( 18 points ) 
# 
# <hr> 
# 
# Now lets take a closer look at the data you have generated. You will want to plot both the function $f(x)$ the data represents, as well as the toy data itself. 
# 
# * Plot the data in a scatter plot with 50% transparency.
# 
# * Plot the function $f(x)$ in a dashed, red line with a linewidth of 4.
# 
# * The title for the plot should be dynamic, as it is passed to the function. Set the fontsize of the title to 17.
# 
# * Label both the scatter plot as 'data' and the line plot as '$f(x)$'.
# 
# * Set the labels of the axes to a fontsize of 14. Those labels are also passed to the plotting function.
# 
# * The label of the y-axis should be rotated 90 degrees.
# 
# 
# * Loops allowed in this excercise: 0

# In[9]:


from typing import Callable, Optional

@max_allowed_loops(0)
def plot_data(
    x: np.ndarray,
    y: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    y_hat: Optional[np.ndarray] = None,
    std: float = 0,
    title: str = "",
    ax=None,
    ylabel: str = "y_values",
    xlabel: str = "x_values",
    show_legend: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 5))

    # YOUR CODE HERE
    #plot carac.
    ax.plot(x , y, 'o', label = "data", alpha = 0.5)
    ax.set_title(title, fontsize = 17)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation = 90)
    
    if f is not None:
        ax.plot(x, f(x), "r--", linewidth = 4, label = "f(x)")
       
        
    # YOUR CODE HERE
    
    if y_hat is not None:
        ax.plot(x, y_hat, "--", color="g", linewidth=3, alpha=0.8, label="$\hat{f}(x)$")
        ax.fill_between(x, y_hat + std, y_hat - std, color="g", alpha=0.1)

    if show_legend:
        ax.legend(fontsize=16)

    return ax


# In[11]:


plot_data(x_tr, y_tr, f, title='Scatter plot of the train data', show_legend=True);


# In[12]:


plot_data(x_te, y_te, f, title='Scatter plot of the test data');


# ### Exercise 1.5: Expected output ( 0 points )
# 
# <hr>
# 
# If you have done all above tasks correctly, the plots should look like this:
# 
# * Compare your plots to the correct ones.
# 
# * This gives you a chance to correct obvious mistakes but do not let yourself get stuck here.
# 
# * There is no code to write in this excercise.
# 
# <img src='./images/scatter_train.png' width=500>
# <img src='./images/scatter_test.png' width=500>

# In[ ]:





# # $\textbf{Exercise 2: } \text{Linear Regression (29 Points})$
#  
# In the second excercise you will write the Linear Regression algorithm to predict the function $\hat{f}(x)$ that fits the data best. The polynomial degree of a function is the greatest exponent of that function. E.g. a function with a polynomial degree of 3 looks like this: $$\hat{f_3}(x) = a + b x^1 + c x^2 + d x^3\quad \mathcal{a,b,c,d} \in \mathbb{R}$$
# 
# <hr>
# 
# ### Exercise 2.1: Calculate the polynomial features of $x$ ( 7 points ) 
# 
# <hr>
# 
# What you want to do first is to calculate $X$ containing the x-values to the power of $0, 1, 2, ...$ up to to the power of $\text{degree}$. These values are later used to find the parameters they need to be multiplied with to compute $\hat{f}(x)$. This `get_X(x, degree)` needs to be adaptable to a varying degree, as sometimes a function with a polynomial degree of 6 fits better and other times a function with a polynomial degree of 2 is sufficient.
# 
# * Compute $X$.
# 
# $$d = \text{degree + 1}$$
# 
# $$\Large{
# X = [x^0, x^1, ...,x^d] \in \mathbb{R}^{(N,d)}, \text{where} \quad x \in \mathbb{R}^{(N,1)}
# }$$
# 
# * `get_X(x, degree)` needs to work dynamically with every degree it is getting passed.
# 
# 
# 
# * Loops allowed in this excercise: 0
# 

# In[13]:


@max_allowed_loops(0)
def get_X(x: np.ndarray, degree: int) -> np.ndarray:
    '''
    Calculate all the powers of every input x up to and including `degree`.
    If the input is x=[1,2,3,10], degree=2 the output is:
    [
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9],
        [1, 10, 100],
    ]
    '''
    assert degree >= 0
    # YOUR CODE HERE
    d = degree + 1
    X = (x[:, None] ** np.arange(d))
    # YOUR CODE HERE
    
    return X


# In[14]:


degree = 3
X_tr = get_X(x_tr, degree)
print(X_tr.shape)

assert X_tr.shape == (400, 4)

np.testing.assert_equal(X_tr[:,0], 1)

for i in range(X_tr.shape[1] - 1):
    np.testing.assert_allclose(X_tr[:,i+1] / X_tr[:, i], x_tr)

    
assert get_X(np.random.rand(10), 20).shape == (10,21)


# ### Exercise 2.2: Find the W containing the parameters ( 10 points ) 
# 
# <hr>
# 
# Next you need to find the parameters of your polynomial function. In the example $\hat{f_3}(x) = a + b x^1 + c x^2 + d x^3\quad \mathcal{a,b,c,d} \in \mathbb{R}$ you would want to find $\mathcal{a,b,c,d}$ so that $\hat{f_3}(x)$ fits your data best. To solve this problem you need $y$ to find a vector $W$ so that $\hat{f}(x) = X W$. The value of $\lambda$ regularizes the variance of the model parameters. 
# 
# * Calculate W:
# 
# $$\Large{
# W = (X^{\top}X + \lambda I_d)^{-1}X^{\top}y}, \quad \text{where}\quad X\in \mathbb{R}^{(N,d)}, \quad I_d \in \{0,1\}^{(d,d)}
# $$
# 
# $$\text{Such that}\quad \forall A : \quad A I = A$$
# 
# $$\text{E.g. }d=3, \quad I_3 = \begin{pmatrix}1 & 0 & 0\\\ 0 & 1 & 0 \\\ 0 & 0 & 1  \end{pmatrix}
# $$ 
# 
# 
# * Loops allowed in this excercise: 0

# In[15]:


@max_allowed_loops(0)
def calc_W(X: np.ndarray, y: np.ndarray, lambd: float) -> np.ndarray:
    """
    Calculate the W vector that fits the data `X` into labels `y`, while
    using the regularization parameter `lambd`
    """
    # YOUR CODE HERE
    #identity matrix
    #Matrix dimension
    dim = X.shape
    N,d = dim
    I = np.eye(d)
    
    
    #W
    
    #inverse
    #@ = np.matmul
    
    
    W = np.linalg.solve(X.T @ X + lambd * I, X.T @ y)
    
    return W


# In[16]:


print(X_tr.shape)

W_tr = calc_W(X_tr, y_tr, 1)
W_tr.shape == (4,)

calc_W(np.random.rand(20,7), np.random.rand(20), 0).shape == (7,)


# ### Exercise 2.3: Compute predictions for the train data ( 2 points ) 
# 
# <hr> 
# 
# Now that you have both the polynomial features $X$ and the parameters $W$ of your function $\hat{f}(x)$ you can compute your prediction of $\hat{y} = \hat{f}(x)$ for any x-value.
# 
# * Compute the predictions for the training data:
# 
# $$\Large{\hat{y} = X W}$$
# 
# 
# * Loops allowed for this exercise: 0

# In[17]:


@max_allowed_loops(0)
def compute_y_hat(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Evaluate the fitted polynomial encoded in W for all values in X.
    """
    # YOUR CODE HERE
    y_hat = X @ W
    # YOUR CODE HERE
    
    return y_hat


# In[18]:


y_hat_tr = compute_y_hat(X_tr, W_tr)

assert len(y_hat_tr.shape) == 1
assert y_hat_tr.shape[0] == X_tr.shape[0]

assert compute_y_hat(np.random.rand(20, 4), np.random.rand(4)).shape == (20,)


# ### Exercise 2.4: Linear Regression Class ( 5+5 Points )
# 
# It is practical to organize the steps above in one Linear Regression class.
# 
# * Define the $\text{LinRegression}$ class and its $\text{train}$ and $\text{predict}$ methods using the functions implemented above.
# * The train function computes and stores $W$. To compute $W$ you will also have to compute $X$.
# * The predict function computes $\hat{y}$ by first computing $X$ and using the $W$ calculated in the train function.
# * If $W$ hasn't been calculated yet, the model is not trained. Throw a `ModelNotTrainedException` in that case.
# 
# 
# * Loops allowed for this exercise: 0

# In[19]:


class ModelNotTrainedException(Exception):
    '''
    Exception to represent when a model has not 
    been trained but it is being used.
    '''


# In[22]:


class LinRegression(object):
    def __init__(self, degree: int, lambd: float = 1) -> None:
        self.W = None
        self.degree = degree
        self.lambd = lambd

    def __eq__(self, other: "LinRegression") -> bool:
        return (
            self.degree == other.degree
            and self.lambd == other.lambd
            and np.allclose(self.W, other.W)
        )

    def train(self, x: np.ndarray, y: np.ndarray) -> "LinRegression":
        """
        Train the linear regression instance, by calculating the
        polynomial features for all elements in `x` using `get_X`,
        calculating the W polynomial using `calc_W` and then
        storing the W vector in the instance member `self.W`.

        The method returns the calling instance.
        """
        # YOUR CODE HERE
        
        #Calculating the polynomial features for all elements in x
        calc_X_tr = get_X(x, self.degree)
        
        #Calculating the W polynomial
        self.W = calc_W(calc_X_tr, y, self.lambd)
        # YOUR CODE HERE
        
        return self

    def predict(self, x) -> np.ndarray:
        """
        Predict the y value for all inputs in x. First the
        polynomial features of all elements in `x` are calculated
        using `get_X`, and then the predictions using `compute_y_hat`.

        If the instance has not been trained, an `ModelNotTrainedException`
        is raised.
        """
        # YOUR CODE HERE
        if self.W is None:
            raise ModelNotTrainedException()
            
        calc_X_pre = get_X(x, self.degree)
        y_hat_pre = compute_y_hat(calc_X_pre, self.W)
        
        return y_hat_pre
        # YOUR CODE HERE
        


# In[23]:


degree = 3
lambd = 0.1
linReg = LinRegression(degree, lambd)
assert linReg.W is None
linRegTrained = linReg.train(x_tr, y_tr)
assert linRegTrained is linReg
assert linReg.W is not None

LinRegression(4, 0).train(
    np.random.rand(20), np.random.rand(20)
).W.shape == (5,)


# In[24]:


linReg = LinRegression(degree, lambd).train(x_tr, y_tr)

y_hat_te = linReg.predict(x_te)

try:
    linReg = LinRegression(degree, lambd).predict(np.random.rand(10))
except ModelNotTrainedException as error:
    "Your code should throw an exception and get here."
else:
    assert False, "Your function should raise a ModelNotTrainedException."

y_hat_te


# # $\textbf{Exercise 3: } \text{Parameter variation (27 Points})$
#  
# In the third excercise you will use everything you have implemented so far to see how good the predictions of $\hat{f}(x)$ are. You will also check what influence varying polynomial degrees and varying lambda values have on the predictions to find the best parameter setting for this dataset.
# 
# <hr>
# 
# ### Exercise 3.1:  Define the objective function ( 3 points ) 
# <hr>
# 
# The objective function of Linear Regression is to minimize the sum of the squares of the difference between $y$ and $\hat{y}$ so that we can find the function $\hat{f}(x)$ that is closest to the true model.
# 
# * Write the objective function:
# 
# $$\Large{\mathcal{L}_{te} = \frac1 2|| y_{te} - \hat{y}_{te} ||_2 = \frac1 2 \sqrt{\sum_{i}(y^{(i)}_{te} - \hat{y}^{(i)}_{te})^2}}
# $$
# 
# 
# * Loops allowed in this exercise: 0

# In[25]:


@max_allowed_loops(0)
def L(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the L2 loss between the predicted and generated values.
    """
    # YOUR CODE HERE
    Loss = 0.5 * np.sum( (y - y_hat)**2) **0.5
    # YOUR CODE HERE
    
    return Loss


# In[26]:


L_te = L(y_te, y_hat_te)


# In[ ]:





# ### Exercise 3.2: Compute the standard deviation ( 4 Points )
# 
# The standard deviation is an interesting measurement to see how different $y$ and $\hat{y}$ are from each other.
# 
# * Compute the standard deviation:
# 
# $$\Large{
# \sigma = \sqrt{\frac1 n \sum_{i=1}^N(y - \hat{y})^2}
# }$$
# * Solve this exercise without using `np.std`.
# 
# 
# * Loops allowed for this exercise: 0

# In[27]:


@max_allowed_loops(0)
def std_(y: np.ndarray, y_hat: np.ndarray) -> float:
    '''
    Calculate the standard deviation of the differences between
    the generated and predicted values.
    '''
    # YOUR CODE HERE
    std = (np.sum((y-y_hat)** 2)/len(y))** 0.5
    # YOUR CODE HERE
    
    return std


# In[28]:


np.testing.assert_allclose(
    np.std(y_tr - y_hat_tr),
    std_(y_tr, y_hat_tr),
    atol=1e-4,
)

print(std_(y_tr, y_hat_tr))


# In[ ]:





# ### Exercise 3.3:  Plotting the test data with different settings ( 15 points )
# 
# <hr>
# 
# With the test data you can now visualize the impact different degrees and lambdas have on the computed function $\hat{f}(x)$.
# 
# * Train a linear regression model for each combination of lambdas and degrees __on the training data__. (The `for` loop is already implemented for you.)
# * Predict $\hat{y}$ __for the test data__.
# * Calculate the L2 loss (using the `L` function) between $y$ and $\hat{y}$.
# * Call the function `plot_data` from Excercise 1.4 and pass `x_te, y_te, f, y_hat, std, title, ax, ylabel, xlabel` and `show_legend` as arguments.
# 
# 
# The settings for every subplot that are passed to `plot_data` are:
# * Calculate the std of the test data.
# * The xlabel is dynamic: '$\lambda$: {lambda}'
# * The ylabel is dynamic: 'degree: {degree}'
# * The title is dynamic: '$\mathcal{L}_{te}: $ {L_te}'
# 
# 
# * Loops allowed in this exercise: 0
# 
# Implementing this function requires a loop for iterating over the parameter combinations, but the part you have to fill in does not require any loops.

# In[29]:


from itertools import product
from typing import Iterable, Tuple


@max_allowed_loops(1)
def plot_test_data_fitting(
    degrees: Iterable[int],
    lambdas: Iterable[float],
    std: float = 0,
    fs: Tuple[int, int] = (20, 20),
    show_legend: bool = False,
):

    fig, axis = plt.subplots(
        len(degrees), len(lambdas), sharex=True, sharey=True, figsize=fs, squeeze=False
    )
    for (n, degree), (k, lambd) in product(enumerate(degrees), enumerate(lambdas)):
        ax = axis[n, k]

        # YOUR CODE HERE
        #Train a linear regression
        
        LR = LinRegression(degree, lambd)
        Train_LR = LR.train(x_tr, y_tr)
        y_hat_te = Train_LR.predict(x_te)
        
        #Calculating loss and sigma
        if std is None or std == 0:
            std = std_(y_te, y_hat_te)
    
        calc_loss = L(y_te, y_hat_te)
        
        #Ploting data
        #a way to combine f-string and latex
        plot_data(x_te, y_te, f, y_hat=y_hat_te, std = std, title = r'$\mathcal{L}_{te}:$' + f'{calc_loss:0.3f}', ax = ax, ylabel = f'degree: {degree}', xlabel = r'$\lambda$'+ f': {lambd}')
        # YOUR CODE HERE
        

    fig.tight_layout()


# In[30]:


degrees = [2, 6, 10, 16, 17]
lambdas = [0, 1, 10]

plot_test_data_fitting(degrees, lambdas)


# ### Exercise 3.4: Expected output ( 0 points )
# 
# <hr>
# 
# If you have done all above tasks correctly, the plots should look like this:
# 
# * Compare your plots to the correct ones.
# 
# * This gives you a chance to correct obvious mistakes but do not let yourself get stuck here.
# 
# * There is no code to write in this excercise.
# 
# <img src='./images/fitting.png' width=1000>

# ### Exercise 3.5: Grid search of the parameters ( 5 points )
# 
# <hr>
# 
# Now that you have a way to rate how far $y$ is from $\hat{y}$ (the objective function $L$), you can try different combinations of degrees and lambdas to find the ideal parameters.
# 
# * Use the functions and classes you have implemented up to this point.
# * Train a linear regression model for each combination of lambdas and degrees __on the training data__.
# * Predict $\hat{y}$ __for the test data__.
# * Calculate the L2 loss (using the `L` function) between $y$ and $\hat{y}$.
# * Return the best loss corresponding model for the best combination of degree and lambda. 
# 
# 
# * Number of loops allowed in this exercise: 0
# 
# Implementing this function requires a loop for iterating over the parameter combinations, but the part you have to fill in does not require any loops.

# In[31]:


degrees = np.arange(2, 20)
lambdas = [0, 1, 10, 100]


# In[39]:


from itertools import product
from typing import Iterable


@max_allowed_loops(1)
def grid_search(
    degrees: Iterable[int],
    lambdas: Iterable[float],
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> Tuple[LinRegression, float]:
    """
    Find the best hyperparameters for a linear regression model. For each
    hyperparameter combination, train the model using the training data
    and calculate the loss using the test data. If the trained model is
    the best one trained so far according to the loss metric, store it.

    Finally return the best linear regression model along with its test
    loss.
    """

    best_L = float("inf")
    best_linReg = None

    for lambd, degree in product(lambdas, degrees):
        # YOUR CODE HERE
        #Calculating Linear Regression and training it
        LR = LinRegression(degree, lambd)
        LR_Trained = LR.train(x_tr, y_tr)
        Y_Pred = LR_Trained.predict(x_te)
        Loss = L(y_te, Y_Pred)
        
        #Testing the best linear regression model and returning it along with its test loss
        if best_L > Loss:
            best_L = Loss
            best_linReg = LR_Trained
        
        # YOUR CODE HERE

    return best_linReg, best_L


# In[38]:


best_linReg, best_L = grid_search(degrees, lambdas, x_tr, y_tr, x_te, y_te)
print(best_L)
assert best_linReg.W is not None
assert 2.5 < best_L < 2.6


# In[34]:


print(f'Best degree: {best_linReg.degree}')
print(f'Best lambda: {best_linReg.lambd}')
print(f'Best L: {best_L:0.3f}')


# In[35]:


plot_test_data_fitting([best_linReg.degree], [best_linReg.lambd], fs=(12,5), show_legend=True)


# In[ ]:





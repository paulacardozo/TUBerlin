#!/usr/bin/env python
# coding: utf-8

# ![](https://www.tu-berlin.de/fileadmin/a70100710_summeruniversity/Summer_Uni_allgemein/Summer_Uni_Logo.png)

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

# In[25]:


import sys

if (3,7) <= sys.version_info[:2] <= (3, 9):
    print("Correct Python version")
else:
    print(f"You are using a wrong version of Python: {'.'.join(map(str,sys.version_info[:3]))}")


# In[26]:


# This cell is for grading. DO NOT remove it

# Use unittest asserts
import unittest; t = unittest.TestCase()
from pprint import pprint


# # Homework 3: Timing, Numpy, Plotting
# 
# In the previous exercise sheet we introduced several methods for classification: decision trees, nearest neighbors, and nearest means. Of those, the one that could learn from the data, and that also offered enough complexity to produce an accurate decision function was k-nearest neighbors. However, nearest neighbors can be slow when implemented in pure Python (i.e. with loops). This is especially the case when the number of data points or input dimensions is large.
# 
# In this exercise sheet, we will speed up nearest neighbors by utilizing `numpy` and `scipy` packages. Your task will be to **replace list-based operations by vector-based operations** between numpy arrays. The speed and correctness of the implementations will then be tested. In particular, performance graphs will be drawn using the library `matplotlib`.
# 
# Make sure to have installed all the required packages (numpy, scipy). For this you can use `conda install <package>` or `pip install <package>`. Make sure that you restart the Jupyter Server after you have installed any packages.

# In[27]:


try:
    import numpy
    import scipy
except ImportError:
    print("Please install NumPy and SciPy using the instructions above.")
else:
    numpy_version = tuple(map(int, numpy.__version__.split(".")))
    scipy_version = tuple(map(int, scipy.__version__.split(".")))
    if numpy_version >= (1, 18, 0):
        print("NumPy version ok!")
    else:
        print("Your NumPy version is too old!!!")

    if scipy_version >= (1, 6, 0):
        print("SciPy version ok!")
    else:
        print("Your SciPy version is too old!!!")
        


# # Warm Ups
# 
# Before starting the homework sheet we recommend you finish these warm-up tasks. They won't get you any points but should help you get familiar with Numpy.

# In[50]:


import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # random one-dimensional integer array
x2 = np.random.randint(10, size=(5, 4))  # random two-dimensional integer array


# ### Shape of an Array (0 P)
# 
# Write a function that returns the number of rows and the number of columns of an array.
# 
# * Use the attribute `.shape` that every numpy array has.

# In[51]:


def array_shape(array):
    
    number_of_rows = array.shape[0]
    if len(array.shape) > 1: #2D or more
        number_of_columns = array.shape[1]
    else:
        number_of_columns = 0 #1D


    return number_of_rows, number_of_columns


# In[52]:


# Test array_shape function
x1_number_of_rows, x1_number_of_columns = array_shape(x1)
x2_number_of_rows, x2_number_of_columns = array_shape(x2)

t.assertEqual(x1_number_of_rows, 6)
t.assertEqual(x1_number_of_columns, 0)
t.assertEqual(x2_number_of_rows, 5)
t.assertEqual(x2_number_of_columns, 4)


# ### Indexing (0 P)
# 
# Return subarrays of the given arrays according to the conditions. Use array indexing e.g. `x1[1:5:-2]` instead of  loops or hardcoding the solutions.
# 
# * Save the second to last element of `x1` in the variable `x1_second_to_last`.
# * Save a subarray that has every other element of `x1` in the variable `x1_every_other_element`.
# * Save a reversed `x1` in the variable `x1_reversed`.
# * Save the element in row 3 and column 2 of `x2` in the variable `x2_element_in_row_3_and_column_2`. Please note that since indexing starts at zero so row 3 is actually the forth row.
# * Save a subarray/matrix that contains rows 2 to 4 and columns 0 to 3 of `x2` in the variable `x2_rows_2_to_4_columns_0_to_3`. In this case row 4 and column 3 should be INCLUDED.
# 
# Try **not** to use the shape or length of an array for this exercise

# In[53]:


x1_second_to_last = x1[-2]
x1_every_other_element = x1[::2]
x1_every_other_element
x1_reversed = x1[::-1]
x2_element_in_row_3_and_column_2 = x2[3, 2]
x2_rows_2_to_4_columns_0_to_3 = x2[2:5, 0:4]


# In[54]:


# Test indexing solutions
t.assertEqual(x1_second_to_last, 7)
np.testing.assert_allclose(x1_every_other_element, np.array((5, 3, 7)))
np.testing.assert_allclose(x1_reversed, np.array((9,7,3,3,0,5)))
t.assertEqual(x2_element_in_row_3_and_column_2, 5)
np.testing.assert_allclose(x2_rows_2_to_4_columns_0_to_3, np.array(((1,6,7,7),(8,1,5,9),(8,9,4,3))))


# ### Broadcasting (0 P)
# 
# Understanding broadcasting is an important part of understanding numpy.
# 
# * Using `np.newaxis`, turn `array_a` into a column-vector and save the result in the variable `array_a_to_column_vector`.
# * Add the one-dimensional `array_a` and the two dimensional `array_b` together. Do not use any function and only the `+` operator.
# * Add the one-dimensional `array_a` and the two dimensional `array_c` together. Now it is important to use broadcasting since the dimensions of the two arrays do not match: `array_a.shape = (3,)` and `array_c.shape = (3,2).` Addition would work if the shape of `array_a` would be `(3,1)`.

# In[55]:


array_a = np.ones(3)
array_b = np.arange(6).reshape((2,3))
array_c = np.arange(6).reshape((3,2))


# In[56]:


array_a_to_column_vector = array_a[:, np.newaxis]
array_a_plus_array_b = array_a + array_b
array_a_plus_array_c = array_a_to_column_vector + array_c


# In[57]:


# Test broadcasting solutions
np.testing.assert_allclose(array_a_to_column_vector, np.ones(3).reshape(3,1))
np.testing.assert_allclose(array_a_plus_array_b, np.array(((1,2,3),(4,5,6))))
np.testing.assert_allclose(array_a_plus_array_c, np.array(((1,2),(3,4),(5,6))))


# In[58]:


# do not use numpy from now on
del np


# ## Python Nearest Neighbor
# 
# The most basic element of computation of nearest neighbors is its distance function relating two arbitrary data points `x1` and `x2`. We assume that these points are iterable (i.e. we can use a loop over their dimensions). One way among others to compute the **square** Euclidean distance between two points is by computing the sum of the component-wise distances.

# In[59]:


def pydistance(x1: 'Vector', x2: 'Vector') -> float:
    '''
    Calculates the square eucledian distance between two data points x1, x2
    
    Args:
        x1, x2 (vector-like): Two vectors (ndim=1) for which we want to calculate the distance
            `len(x1) == len(x2)` will always be True
        
    Returns: 
        float: The square eucleadian distance between the two vectors
    '''
    return sum([(x1d - x2d) ** 2 for x1d, x2d in zip(x1, x2)])


# In[60]:


x1, x2 = [1, 4, 3, 2], [4, 8, -2, 2]
print(f'pydistance({x1}, {x1}) --> {pydistance(x1, x1)}')
print(f'pydistance({x1}, {x2}) --> {pydistance(x1, x2)}')


# where we use the prefix "`py-`" of the function to indicate that the latter makes use of pure `Python` instead of `numpy`. Once the distance matrix has been implemented, the nearest neighbor for a given unlabeled point `u` that we would like to classify is obtained by iterating over all points in the training set `(X, Y)`, selecting the point with smallest distance to `u`, and returning its corresponding label. Here `X` denotes the list of inputs in the training set and `Y` denotes the list of labels.

# In[61]:


def pynearest(u: list, X: list, Y: list, distance: callable = pydistance) -> int:
    '''
    Applies the nearest neighbour to the input `u`
    with training set `X` and labels `Y`. The 
    distance metric can be specified using the
    `distance` argument.
    
    Args:
        u (list): The input vector for which we want a prediction
        X (list): A 2 dimensional list containing the trainnig set
        Y (list): A list containing the labels for each vector in the training set
        distance (callable): The distance metric. By default the `pydistance` function
        
    Returns: 
        int: The label of the closest datapoint to u in X
    '''
    xbest = None
    ybest = None
    dbest = float('inf')
    
    for x, y in zip(X, Y):
        d = distance(u, x)
        if d < dbest:
            ybest = y
            xbest = x
            dbest = d
            
    return ybest


# Note that this function either uses function `pydistance` (given as default if the argument distance is not specified). Or one could specify as argument a more optimized function for distance compuation, for example, one that uses `numpy`. Finally, one might not be interested in classifying a single point, but many of them. The method below receives a collection of such unlabeled test points stored in the variable `U`. The function returns a list of predictions associated to each test point.

# In[62]:


def pybatch(U, X, Y, nearest=pynearest, distance=pydistance):
    '''
    Applies the nearest neighbor algorithm, to all the datapoints
    `u` $\in$ `U`, with `X` the training set and `Y` the labels.
    Both the distance metric and the method of finding the 
    neearest neighbor can be specified.
    
    Args:
        U (list): List of vectors for which a prediction is desired.
        X (list): A 2 dimensional list containing the trainnig set
        Y (list): A list containing the labels for each vector in the training set
        nearest (callable): The method by which the nearest neighbor search happens.
        distance (callable): The distance metric. By default the `pydistance` function
        
    Returns: 
        list: A list of predicted labels for each `u` $\in$ `U`
    '''
    return [nearest(u, X, Y, distance=distance) for u in U]


# Again, such function uses by default the Python nearest neighbor search (with a specified distance function). However, we can also specified a more optimized nearest neighbor function, for example, based on `numpy`. Finally, one could consider an alternative function to `pybatch` that would use `numpy` from the beginning to the end. The implementation of such more optimized functions, and the testing of their correct behavior and higher performance will be the objective of this exercise sheet.

# ## Testing and correctness
# 
# As a starting point, the code below tests the output of the nearest neighbor algorithm for some toy dataset with fixed parameters. In particular, the function `data.toy(M,N,d)` generates a problem with `M` unlabeled test points stored in a matrix `U` of size `(M x d)`, then `N` labeled training points stored in a matrix `X` of size `(N x d)` and the output label is stored in a vector `Y` of size `N` composed of zeros and ones encoding the two possible classes. The variable `d` denotes the number of dimensions of each point. The toy dataset is pseudo-random, that is, for fixed parameters, it produce a random-looking dataset, but every time the method is called with the same parameters, the dataset is the same. The pseudo-randomness property will be useful to verify that each nearest neighbor implementation performs the same overall computation. Please check the `data.py` file within the exercise folder for the implementation details. 

# In[63]:


import os
if 'data.py' not in os.listdir():
    t.fail('Did you download the \'data.py\' file from ISIS?')
    

import data
U, X, Y = data.toy(20, 100, 50)

print(f'Shape of U (unlabeled datapoints): {U.shape}')
print(f'Shape of X (training set): {X.shape}')
print(f'Shape of Y (labels): {Y.shape}')
print(f'Predictions: {pybatch(U, X, Y)}')


# In particular, the output of this function will help us to verify that the more optimized `numpy`-based versions of nearest neighbor are still valid.

# ## Plotting and performance
# 
# We now describe how to build a plot that relates a certain parameter of the dataset (e.g. the number of input dimensions `d` to the time required for the computation. We first initialize the basic plotting environment.

# In[64]:


import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 90


# The command "`%matplotlib inline`" tells IPython notebook that the plots should be rendered inside the notebook. 
# 
# The following code plots the computation time of predicting `100` points from the test set using a training set of size `100`, and where we vary the number of input dimensions. The measurement of time happens with the `timeit` module. `timeit` provides many convinience functions for benchmarking. In particular the repeat function runs the provided code many times and returns the time it took to run it. You can find more information about `repeat` [here](https://docs.python.org/3/library/timeit.html#timeit.repeat)

# In[66]:


import timeit
from statistics import mean

# Values for the number of dimensions d to test
dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# Measure the computation time for each choice of number of dimensions d
tlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y), number=1, repeat=3))
    tlist.append(delta)

# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlist, tlist, '-o')
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)


# The time on the vertical axis is in seconds. Note that the exact computation time depends on the speed of your computer. As expected, the computation time increases with the number of input dimensions. Unfortunately, for the small dataset considered here (`100` training and test points of `100` dimensions each), the algorithm already takes more than one second to execute. Thus, it is necessary for practical applications (e.g. the digit recognition task that we will consider at the end of this exercise sheet) to accelerate this nearest neighbor algorithm.

# ## 1. Accelerating the distance computation (25 P)
# 
# In this first exercise, we would like to accelerate the function that compute pairwise distances.
# 
# **a)** Implement the function `npdistance(x1,x2)` with the same output as `pydistance(x1,x2)`, but that computes the squared Euclidean distance using `numpy` operations. Verify that in both cases (i.e. using either `npdistance` or `pydistance` in the function `pybatch`) the output for the above toy example with parameters `M=20`, `N=100`, `d=50` (i.e. `data.toy(20,100,50)`) remains the same.
# 
# Our goal with this exercise is to speed-up our code. In practice this means that we want to remove for loops from our code. Therefore if your implementation contains a `for loop` it will automatically be considered wrong and will receive 0 points. Similarlly Python functions that hide for loops such as `map` are also considered invalid for this exercise. Similarly, functions provided by numpy that hide for loops like [`vectorize`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html) and [`apply_along_axis`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html) are also **not** to be used.
# 
# **Note**: The input vectors can be either `np.ndarray` or lists of `floats`.

# In[67]:


import numpy as np

def npdistance(x1: 'vector-like', x2: 'vector-like') -> float:
    '''
    Calculates the square eucledian distance between two data points x1, x2
    using `numpy` vectorized operations
    
    Args:
        x1, x2 (vector-like): Two vectors (ndim=1) for which we want to calculate the distance
            `len(x1) == len(x2)` will always be True
    
    Returns: 
        float: The distance between the two vectors x1, x2
    '''
    p1 = np.array(x1)
    p2 = np.array(x2)
    dist = np.linalg.norm(p1 - p2)**2
    
    return dist
    


# In[68]:


# Verify your function
x1, x2 = [0.,-1.,-2.], [2.,3.,4.]

dist_to_same = npdistance(x1, x1)
print(f'npdistance({x1}, {x1}) --> {dist_to_same}\n')
expected_dist_to_same = 0.
t.assertAlmostEqual(dist_to_same, expected_dist_to_same, 
                    msg='The distance of a vector to itself should be 0')


dist = npdistance(x1, x2)
print(f'npdistance({x1}, {x2}) --> {dist}')
expected_dist = pydistance(x1, x2)
print(f'expected_dist --> {expected_dist}\n')
t.assertAlmostEqual(dist, expected_dist)

U, X, Y = data.toy(20,100,50)

no_numpy = pybatch(U, X, Y, distance=pydistance)
print(f'no_numpy --> {no_numpy}')

w_np_dist = pybatch(U, X, Y, distance=npdistance)
print(f'w_np_dist  --> {w_np_dist}')

np.testing.assert_allclose(no_numpy, w_np_dist)


# **b)** Create a plot similar to the one above, but where the computation time required by both methods are shown in a superposed manner. Here, we fix `M=100`, `N=100`, and we let `d` vary from `1` to `500`, taking the list of values `[1, 2, 5, 10, 20, 50, 100, 200, 500]`. Your plot should show a quisi-constant runtime for the pybarch call using the `npdistance` function, compared to `pydistance`.

# In[46]:


import timeit
from statistics import mean

# Values for the number of dimensions d to test
dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# Measure the computation time for each choice of number of dimensions d
#Numpy
nptlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y, distance = npdistance), number=1, repeat=3))
    nptlist.append(delta)

#Python
pytlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y), number=1, repeat=3))
    pytlist.append(delta)
    
# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlist, nptlist, '-o', label = "numpy")
plt.plot(dlist, pytlist, '-o', label = "python")
legen = plt.legend(loc='best', fontsize=10)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)


# **c)** Based on your results, explain what kind of speedup `numpy` provides, and in what regime do you expect the speedup to be the most important:
# 
# **Note**: For this exercise you only need to provide a free text answer
# 

# #### Explain the speedup that numpy provides
# 
# 

# ## 2. Accelerating the nearest neighbor search (25 P)
# 
# Motivated by the success of the `numpy` optimized distance computation, we would like further accelerate the code by performing nearest neighbor search directly in `numpy`.
# 
# **a)** Implement the function `npnearest(u,X,Y)` as an alternative to the function `pynearest(u,X,Y,distance=npdistance)` that we have used in the previous exercise. Again, verify your function for the same toy example as before (i.e. `data.toy(20,100,50)`).
# 
# Unlike `pynearest`, `npnearest` doesn't receive any distance argument. `npnearest` will work only with square eucledian distance. If you are confident that your `npdistance` implementation can work between a vector and a matrix, you are welcome to reuse it. It is however, perfectly acceptable to reimplement the distance algorithm in this function again.
# 
# Once again the use of `for loops`, or functions like `map` or `vectorize` is stictly not allowed in this exercise.

# In[47]:


def npnearest(u: np.ndarray, X: np.ndarray, Y: np.ndarray, *args, **kwargs):
    '''
    Finds x1 so that x1 is in X and u and x1 have a minimal distance (according to the 
    provided distance function) compared to all other data points in X. Returns the label of x1
    
    Args:
        u (np.ndarray): The vector (ndim=1) we want to classify
        X (np.ndarray): A matrix (ndim=2) with training data points (vectors)
        Y (np.ndarray): A vector containing the label of each data point in X
        args, kwargs  : Ignored. Only for compatibility with pybatch
        
    Returns:
        int: The label of the data point which is closest to `u`
    '''
    dis = npdistance(u, X) 
    index = dis.argmin() #works like flatten, returns the index of the minimum value
    l = Y[index]
    
    return l
    


# In[69]:


TINY_U, TINY_X, TINY_Y = data.toy(3,3,3)
tiny_u = TINY_U[0]
print('u')
pprint(tiny_u)
print('\nX')
pprint(TINY_X)
print('\nY')
pprint(TINY_Y)

np_nearest = npnearest(tiny_u, TINY_X, TINY_Y)
expected_nearest = pynearest(tiny_u, TINY_X, TINY_Y)
print(f'\nnp_nearest --> {np_nearest}')
print(f'expected_nearest --> {expected_nearest}')

t.assertEqual(expected_nearest, np_nearest)

# Verify your function
np.testing.assert_allclose(
    pybatch(U, X, Y, nearest=pynearest), 
    pybatch(U, X, Y, nearest=npnearest)
)


# In[ ]:


# This cell is for grading. DO NOT remove it


# **b)** Create a plot similar to the one above, where the new method is compared to the previous one. This means that you should compare the runtime of `npnearest` and `pynearest` with `npdistance` as its distance function. Here, we fix `M=100`, `d=100`, and we let `N` take different values `[1, 2, 5, 10, 20, 50, 100, 200, 500]`.

# In[ ]:


import timeit
from statistics import mean

# Values for the number of dimensions d to test
dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# Measure the computation time for each choice of number of dimensions d
#npdistance
np_d_tlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y, distance = npdistance), number=1, repeat=3))
    np_d_tlist.append(delta)

#Pynearest
pytlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = pynearest), number=1, repeat=3))
    pytlist.append(delta)
    
#npnearest
np_n_tlist = []
for d in dlist:
    U, X, Y = data.toy(100, 100, d)  
    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = npnearest), number=1, repeat=3))
    np_n_tlist.append(delta)
    
# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlist, np_d_tlist, '-o', label = "npdistance")
plt.plot(dlist, np_n_tlist, '-o', label = "npnearest")
plt.plot(dlist, pytlist, '-o', label = "pynearest")
legen = plt.legend(loc='best', fontsize=10)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)


# ## 3. Accelerating the processing of multiple test points (25 P)
# 
# Not yet fully happy with the performance of the algorithm, we would like to further optimize it by avoiding performing a loop on the test points, and instead, classify them all at once.
# 
# **a)** Implement the function `npbatch(U,X,Y)` as a replacement of the implementation `pybatch(U,X,Y,nearest=npnearest)` that we have built in the previous exercise. Inside this function, use [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) for the actual distance computation. Again, verify your function for the same toy example as before (i.e. `data.toy(20,100,50)`).

# In[ ]:


import scipy

# for some systems we need to import spatial explicitly
from scipy.spatial import distance
def npbatch(U, X, Y, *args, **kwargs):
    '''
    This function has the same functionality as the `pybatch` function.
    HOWEVER, the distance function is fixed (scipy.spatial.distance.cdist).
    It does not use any of the functions defined by us previously.
    
    Args:
        U (np.ndarray): A matrix (ndim=2) containing multiple vectors which we want to classify
        X (np.ndarray): A matrix (ndim=2) that represents the training data
        Y (np.ndarray): A vector (ndim=1) containing the labels for each data point in X
        
        All other arguments are ignored. *args, **kwargs are only there for compatibility 
        with the `pybatch` function
        
    Returns:
        np.ndarray: A vector (ndim=1) with the predicted label for each vector $u \in U$
    '''
    
    d = scipy.spatial.distance.cdist(U, X, metric = 'euclidean') #both are 2D
    
    index = np.argmin(d, axis=1) 
    label = Y[index]
    
    return label
    
    
    
    


# In[ ]:


print('U')
pprint(TINY_U)
print('\nX')
pprint(TINY_X)
print('\nY')
pprint(TINY_Y)

expected_output = pybatch(TINY_U, TINY_X, TINY_Y)
print(f'\nexpected_output --> {expected_output}')
actual_output = npbatch(TINY_U, TINY_X, TINY_Y)
print(f'actual_output --> {actual_output}')
np.testing.assert_allclose(expected_output, actual_output)

U, X, Y = data.toy(20,100,50)
np.testing.assert_allclose(pybatch(U, X, Y), npbatch(U, X, Y))


# In[ ]:


# This cell is for grading. DO NOT remove it


# **b)** Create a plot comparing the computation time of the new implementation compared to the previous one. Here, we fix `N=100`, `d=100`, and we let `M` vary from `1` to `500` with values `[1, 2, 5, 10, 20, 50, 100, 200, 500]`.

# In[ ]:


import timeit
from statistics import mean

# Values for the number of dimensions d to test
dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# Measure the computation time for each choice of number of dimensions d
#npdistance
  
#npnearest
np_n_tlist = []
#npbatch
np_b_tlist = []

for d in dlist:
    U, X, Y = data.toy(100, d, 100)  

    # get the average of three runs
    delta = mean(timeit.repeat(lambda : pybatch(U,X,Y, nearest = npnearest), number=1, repeat=3))
    np_n_tlist.append(delta)
    

    # get the average of three runs
    delta = mean(timeit.repeat(lambda : npbatch(U,X,Y), number=1, repeat=3))
    np_b_tlist.append(delta)
    
# Plot the results in a graph
fig = plt.figure(figsize=(5, 3))
plt.plot(dlist, np_b_tlist, '-o', label = "npbatch")
plt.plot(dlist, np_n_tlist, '-o', label = "npnearest")
legen = plt.legend(loc='best', fontsize=10)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)


# In[ ]:





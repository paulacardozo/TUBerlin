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

# In[1]:


import sys

if (3,7) <= sys.version_info[:2] <= (3, 9):
    print("Correct Python version")
else:
    print(f"You are using a wrong version of Python: {'.'.join(map(str,sys.version_info[:3]))}")


# In[2]:


# This cell is for grading. DO NOT remove it

# Use unittest asserts
import unittest

t = unittest.TestCase()
from pprint import pprint

from typing import Tuple, List

# Helper assert function
def assert_percentage(val):
    t.assertGreaterEqual(val, 0.0, f"Percentage ({val}) cannot be < 0")
    t.assertLessEqual(val, 1.0, f"Percentage ({val}) cannot be > 1")


# In[3]:


def decision(x: Tuple[str, int, str]) -> str:
    smoker, age, diet = x
    if smoker == "yes":
        return "less" if age < 29.5 else "more"
    else:
        return "less" if diet == "good" else "more"

def gettest() -> List[Tuple[str, int, str]]:
    return [
        ("yes", 21, "poor"),
        ("no", 50, "good"),
        ("no", 23, "good"),
        ("yes", 45, "poor"),
        ("yes", 51, "good"),
        ("no", 60, "good"),
        ("no", 15, "poor"),
        ("no", 18, "good"),
    ]


def gettrain() -> List[Tuple[Tuple[str, int, str], str]]:
    return [
        (("yes", 54, "good"), "less"),
        (("no", 55, "good"), "less"),
        (("no", 26, "good"), "less"),
        (("yes", 40, "good"), "more"),
        (("yes", 25, "poor"), "less"),
        (("no", 13, "poor"), "more"),
        (("no", 15, "good"), "less"),
        (("no", 50, "poor"), "more"),
        (("yes", 33, "good"), "more"),
        (("no", 35, "good"), "less"),
        (("no", 41, "good"), "less"),
        (("yes", 30, "poor"), "more"),
        (("no", 39, "poor"), "more"),
        (("no", 20, "good"), "less"),
        (("yes", 18, "poor"), "less"),
        (("yes", 55, "good"), "more"),
    ]


# # Homework 2: Further Python Exercises
# 
# In this exercise sheet we will keep building up our Python knowledge. In this homework sheet we will consider data oriatned approaches towards solving the health classification problem that was introduced in the previous homework sheet.
# 

# ## Nearest neighbor classifier (25 P)
# 
# We consider the nearest neighbor algorithm that classifies test points following the label of the nearest neighbor in the training data. You can read more about Nearest neighbor classifiers [here](http://www.robots.ox.ac.uk/~dclaus/digits/neighbour.htm). For this, we need to define a distance function between data points. We define it to be
# 
# `distance(a, b) = (a[0] != b[0]) + ((a[1] - b[1]) / 50.0) ** 2 + (a[2] != b[2])`
# 
# where `a` and `b` are two tuples representing two patients.
# 
# * Implement the distance function.
# * Implement the function that retrieves for a test point the nearest neighbor in the training set, and classifies the test point accordingly (i.e. returns the label of the nearest data point).
# 
# **Hint**: You can use the special `infinity` floating point value with `float('inf')`
# 
# ***Keep in mind that `bool`s in Python are also `int`s. `True` is the same as `1` and `False` is the same as `0`***

# In[4]:


def distance(a: Tuple[str, int, str], b: Tuple[str, int, str]) -> float:
    """
    Calculates the distance between two data points (patient tuples)
    Args:
        a, b (Tuple[str, int, str]): Two patient tuples for which we want to calculate the distance
    Returns:
        float: The distance between a, b according to the above formula
    """
    
    outp = (a[0] != b[0]) + ((a[1] - b[1]) / 50.0)** 2 + (a[2] != b[2])
    return outp


# In[5]:


# Test distance
x1 = ("yes", 34, "poor")
x2 = ("yes", 51, "good")
dist = distance(x1, x2)
print(f"distance({x1}, {x2}) --> {dist}")
expected_dist = 1.1156
t.assertAlmostEqual(dist, expected_dist)


# In[6]:


# This cell is for grading. DO NOT remove it


# In[7]:


def neighbor(
    x: Tuple[str, int, str],
    trainset: List[Tuple[Tuple[str, int, str], str]],
) -> str:
    """
    Returns the label of the nearest data point in trainset to x.
    If x is `('no', 30, 'good')` and the nearest data point in trainset
    is `('no', 31, 'good')` with label `'less'` then `'less'` will be returned.
    In case two elements have the exact same distance, element that first occurs
    in the dataset is picked (the element with the smallest index).

    Args:
        x (Tuple[str, int, str]): The data point for which we want
        to find the nearest neighbor
        trainset (List[Tuple[Tuple[str, int, str], str]]):
        A list of tuples with patient tuples and a label

    Returns:
        str: The label of the nearest data point in the trainset.
        Can only be 'more' or 'less'
    """
    d_min = float('inf') #value for comparison
    min_label = None
    
    for y in trainset:
        features, label = y
        d = distance(x, features)
        
        if d < d_min:
            d_min = d
            min_label = label
            
    return min_label
            
        
    


# In[8]:


# Test neighbor
x = ("yes", 31, "good")
prediction = neighbor(x, gettrain())
print(f"prediction --> {prediction}")
expected = "more"
t.assertEqual(prediction, expected)


# In[9]:


# This cell is for grading. DO NOT remove it


# In this part we want to compare the decision tree we have implemented with the nearest neighbor method. Apply both the decision tree and nearest neighbor classifiers on the test set, and return the list of data point(s) for which the two classifiers disagree, and with which probability it happens.

# In[10]:


def compare_classifiers(
    trainset: List[Tuple[Tuple[str, int, str], str]],
    testset: List[Tuple[str, int, str]],
) -> Tuple[List[Tuple[str, int, str]], float]:
    """
    This function compares the two classification methods (decision tree, nearest neighbor)
    by finding all the datapoints for which the methods disagree. It returns
    a list of the test datapoints for which the two methods do not return
    the same label as well as the ratio of those datapoints compared to the whole
    test set.

    Args:
        trainset (List[Tuple[Tuple[str, int, str], str]]):
        The training set used by the nearest neighbour classfier.
        testset (List[Tuple[str, int, str]]): Contains the elements
        which will be used to compare the decision tree and nearest
        neighbor classification methods.

    Returns:
        Tuple[List[Tuple[str, int, str]], float]: A list containing all the data points which yield
        different results for the two classification methods. The ratio of
        datapoints for which the two methods disagree.

    """
    disagree = []
    cnt = 0
    for element in testset:
        if neighbor(element, trainset) != decision (element):
            disagree.append(element)
            cnt += 1
            
    percentage = cnt / len(testset)
            
    return disagree, percentage


# In[11]:


# Test compare_classifiers
disagree, ratio = compare_classifiers(gettrain(), gettest())
print(f"ratio = {ratio}")
t.assertIsInstance(disagree, list)
t.assertIsInstance(ratio, float)
t.assertIsInstance(disagree[0], tuple)
t.assertEqual(len(disagree[0]), 3)
assert_percentage(ratio)
t.assertTrue(0.1 < ratio < 0.2)


# One problem of simple nearest neighbors is that one needs to compare the point to predict to all data points in the training set. This can be slow for datasets of thousands of points or more. Alternatively, some classifiers train a model first, and then use it to classify the data.
# 
# ## Nearest mean classifier (25 P)
# 
# We consider one such trainable model, which operates in two steps:
# 
# 1. Compute the average point for each class
# 2. Classify new points to be of the class whose average point is nearest to the point to predict.
# 
# For this classifier, we convert the attributes smoker and diet to real values (for smoker: `1.0` if 'yes' otherwise `0.0`, and for diet: `0.0` if 'good' otherwise `1.0`), and use the modified distance function:
# 
# `distance(a,b) = (a[0] - b[0]) ** 2 + ((a[1] - b[1]) / 50.0) ** 2 + (a[2] - b[2]) ** 2`
# 
# Age will also from now on be represented as a `float`. The new data points will be referred to as numerical patient tuples. 
# 
# We adopt an object-oriented approach for building this classifier.
# 
# * Implement the `gettrain_num` function that will load the training dataset from the `health-train.txt` file and parse each line to a numerical patient tuple with its label. You can still follow the same structure that we used before (i.e. using a `parse_line_...` function), however, it is not required for this exercise. Only the `gettrain_num` function will be tested.
# 
# 
# * Implement the new distance function.
# 
# 
# * Implement the methods `train` and `predict` of the class `NearestMeanClassifier`.

# In[12]:


def parse_line_train_num(
    line: str,
) -> Tuple[Tuple[float, float, float], str]:
    """
    Takes a line from the file `health-train.txt`, including a newline,
    and parses it into a numerical patient tuple.

    Args:
        line (str): A line from the `health-test.txt` file
    Returns:
        Tuple[Tuple[float, float, float], str]:
        A numerical patient tuple.
    """

    smoker, age, diet, label = line.strip().split(",")
    if smoker == "yes":
        smoker = 1.0
    else:
        smoker = 0.0
        
    if diet == "good":
        diet = 0.0
    else:
        diet = 1.0
    
    return (smoker, float(age), diet), label 
    
    #smoker, age, diet, label = line.strip().strip(",")
    #yes = 1.0
    #no = 0.0
    #good = 0.0
    #otherwise = 1.0
    #return ( (smoker, float(age), diet), label)
    
    #entry = line.strip().strip(",")
    #if entry[0] == "yes":
        #entry[0] = 1.0
    #else:
        #entry[0] = 0
    
    


def gettrain_num() -> List[Tuple[Tuple[float, float, float], str]]:
    """
    Parses the `health-train.txt` file into numerical patient tuples.

    Returns:
        List[Tuple[Tuple[float, float, float], str]]:
        A list of tuples containing numerical patient tuples and their labels
    """
    data = []
    with open('health-train.txt', 'r') as file:
        for line in file:
            print(line)
            num = parse_line_train_num(line)
            data.append(num)
    
    return data
    


# In[13]:


# Test gettrain_num
trainset_num = gettrain_num()
t.assertIsInstance(trainset_num, list)
first_datapoint = trainset_num[0]
print(f"first_datapoint --> {first_datapoint}")
t.assertIsInstance(first_datapoint[0], tuple)
t.assertIsInstance(first_datapoint[0][0], float)
t.assertIsInstance(first_datapoint[0][1], float)
t.assertIsInstance(first_datapoint[0][2], float)


# In[14]:


# This cell is for grading. DO NOT remove it


# In[15]:


def distance_num(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """
    Calculates the distance between two numerical patient tuples.
    Args:
        a, b (Tuple[float, float, float]): Two numerical patient tuples for which
        we want to calculate the distance.
    Returns:
        float: The distance between a, b.
    """

    d = (a[0] - b[0]) ** 2 + ((a[1] - b[1]) / 50.0) ** 2 + (a[2] - b[2]) ** 2
    return d
    


# In[16]:


x1 = (1.0, 23.0, 0.0)
x2 = (0.0, 41.0, 1.0)
dist = distance_num(x1, x2)
print(f"dist --> {dist}")
t.assertIsInstance(dist, float)
t.assertTrue(2.12 < dist < 2.13)


# In[17]:


# This cell is for grading. DO NOT remove it


# In[18]:


class NearestMeanClassifier:
    """
    Represents a NearestMeanClassifier.

    When an instance is trained a dataset is provided and the mean for each class is calculated.
    During prediction the instance compares the datapoint to each class mean (not all datapoints)
    and returns the label of the class mean to which the datapoint is closest to.

    Instance Attributes:
        more (Tuple[float, float, float]): A tuple representing the mean of
        every 'more' data-point in the dataset
        less (Tuple[float, float, float]): A tuple representing the mean of
        every 'less' data-point in the dataset
    """

    def __init__(self):
        self.more: Tuple[float, float, float]
        self.less: Tuple[float, float, float]

    def train(
        self,
        dataset: List[Tuple[Tuple[float, float, float], str]],
    ):
        """
        Calculates the class means for a given dataset and stores
        them in instance attributes more, less.

        The mean of the more class is a tuple containing three elements.
        Each element of the mean tuple contains the mean of all the elements
        in the training set that are labeled `more` for each corresponding index.
        This means that the mean tuple contains the mean smoker, age and health
        values.
        The same is true of the less mean tuple, but for all the elements
        labeled `less`.

        This function is does not return anything useful, but it has the side
        effect of setting the more and less instance variables.

        Args:
            dataset (List[Tuple[Tuple[float, float, float], str]]):
            A list of tuples each of them containing a numerical patient tuple and its label
        Returns:
            self
        """
        #More and less list.
        
        M = 0
        L = 0
        
        smoker_m = []
        age_m = []
        diet_m = []
        
        smoker_L = []
        age_L = []
        diet_L = []
        
        for datapoint in dataset:
            # (1, 25, 0), "more"
            f, label = datapoint
            
            if label == "more":
                M += 1
                smoker_m += [f[0]]
                age_m.append(f[1])
                diet_m += [f[2]]
                    
            else:
                L += 1
                smoker_L += [f[0]]
                age_L += [f[1]]
                diet_L += [f[2]]
                    
        #Final mean
        smoker_mean_m = sum(smoker_m)/M
        age_mean_m = sum(age_m)/M
        diet_mean_m = sum(diet_m)/M
        smoker_mean_L = sum(smoker_L)/L
        age_mean_L = sum(age_L)/L
        diet_mean_L = sum(diet_L)/L
        
        #Self
        
        self.more = (smoker_mean_m, age_mean_m, diet_mean_m)
        self.less = (smoker_mean_L, age_mean_L, diet_mean_L)
        
        
        
        return self

    def predict(self, x: Tuple[float, float, float]) -> str:
        """
        Returns a prediction/label for numeric patient tuple x.
        The classifier compares the given data point to the mean
        class tuples of each class and returns the label of the
        class to which x is the closest to (according to our
        distance function).

        Args:
            x (Tuple[float, float, float]): A numerical patient tuple
            for which we want a prediction

        Returns:
            str: The predicted label.
        """
        
        if distance_num(self.less, x) >= distance_num(self.more, x):
            prediction = "more"
        else:
            prediction = "less"
            
        return prediction
    
    def __repr__(self):
        try:
            more = tuple(round(m, 3) for m in self.more)
            less = tuple(round(l, 3) for l in self.less)
        except AttributeError:
            more, less = None, None
        return f"{self.__class__.__name__}(more: {more}, less: {less})"


# * Instantiate the `NearestMeanClassifier`, train it on the training data, and return it

# In[19]:


def build_and_train(
    trainset_num: List[Tuple[Tuple[float, float, float], str]]
) -> NearestMeanClassifier:
    """
    Instantiates the `NearestMeanClassifier`, trains it on the
    `trainset_num` dataset and returns it.

    Args:
        trainset_num (List[Tuple[Tuple[float, float, float], str]]): A list of numerical
        patient tuples with their respective labels

    Returns:
        NearestMeanClassifier: A NearestMeanClassifier trained on `trainset_num`
    """
    classifier = NearestMeanClassifier()
    classifier.train(trainset_num)
    
        
    return classifier
    


# In[20]:


# Test build_and_train
classifier = NearestMeanClassifier()
classifier.train(trainset_num)
t.assertIsInstance(classifier, NearestMeanClassifier)
print(classifier)
try:
    classifier.more, classifier.less
except AttributeError:
    t.fail(
        "Did you train the classifier?"
        " Did you store the mean vector for the 'more' class?"
        " Did you store the mean vector for the 'less' class?",
    )

t.assertIsInstance(classifier.more, tuple)
t.assertIsInstance(classifier.less, tuple)

t.assertEqual(round(classifier.more[1]), 37)
t.assertEqual(round(classifier.less[1]), 32)


# In[21]:


# This cell is for grading. Do NOT remove it


# * Load the test dataset into memory as a list of numerical patient tuples
# * Predict the test data using the nearest mean classifier and return the index of all test examples for which all three classifiers (decision tree, nearest neighbor and nearest mean) agree.
# 
# **Note**: Be careful that the `NearestMeanClassifier` expects the dataset in a different form, compared to the other two methods.

# In[22]:


def gettest_num() -> List[Tuple[float, float, float]]:
    """
    Parses the `health-test.txt` file into numerical patient tuples.

    Returns:
        list: A list containing numerical patient tuples, loaded from `health-test.txt`
    """

    data = []
    with open('health-test.txt', 'r') as file:
        for line in file:
            smoker, age, diet = line.strip().split(',')
            smoker = 1.0 if smoker == "yes" else 0.0
            diet = 0.0 if diet == "good" else 1.0
            data.append((smoker, float(age), diet))

    
    return data


# In[23]:


testset_num = gettest_num()
pprint(testset_num)
t.assertIsInstance(testset_num, list)
t.assertEqual(len(testset_num), 8)
t.assertIsInstance(testset_num[0], tuple)
t.assertEqual(len(testset_num[0]), 3)


# In[28]:


def predict_test() -> List[int]:
    """
    Classifies the test set using all the methods that were developed in this exercise sheet,
    namely `decision`, `neighbor` and `NearestMeanClassifier`.

    This functions loads all the needed data by calling the corresponding functions.
    (gettrain, gettest, gettrain_num, gettest_num)

    Returns:
        List[int]: a list of the indices of all the datapoints for which all three
        classfiers have the same output.
    """
    
    
    train_num = gettrain_num()
    test_num = gettest_num()
    
    agreed_samples = []
    
    nmc = build_and_train(train_num)
    
    for i in test_num: 
        d = decision(i)
        n = neighbor(i, train_num)
        m = nmc.predict(i)
        if  d == n:
            if n == m:
                agreed_samples.append(i)
                
    return agreed_samples


# In[ ]:





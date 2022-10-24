#!/usr/bin/env python
# coding: utf-8

# ![](summer_uni.png)

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


# # Warm Ups
# 
# Before starting the homework sheet we recommend you finish these warm-up tasks. They won't bring any points but should help you to get familiar with Python code.

# ### Function and types (0 P)
# 
# Write a function using list comprehension that returns the types of list elements.
# 
# * The function should be called `types_of`
# * The function expects a list as an input argument.
# * The function should return a list with the types of the given list elements.
# * Read the testing cell to understand how `types_of` is supposed to work.

# In[3]:


def types_of(i):
        lis = []
        for element in i:
            t = type(element)
            lis.append(t)
  
        return lis
    


# In[4]:


# Test type_of_two function
types = types_of([7, 0.7, "hello", True, (2, "s")])

assert isinstance(types, list)
t.assertEqual(types[0], int)
t.assertEqual(types[1], float)
t.assertEqual(types[2], str)
t.assertEqual(types[3], bool)
t.assertEqual(types[-1], tuple)


# ### Concatenation and enumerate (0 P)
# 
# 
# Concatenate the strings from the array 'animals' into one string.
# 
# * Use: `counting +=` and string formatting (`f-strings`).
# * Use `enumerate` to get the `i`th index.
# * The result should look as follows: `'0: mouse | 1: rabbit | 2: cat | 3: dog |'`
# 
# ***Note that this is not the most efficient way to concatenate strings in Python but part of this exercise is to showcase `for-loops`***

# In[5]:


animals = ["mouse", "rabbit", "cat", "dog"]


# In[6]:


counting = "|"
for i, animal in enumerate(animals):
      counting += f"{i}: {animal} |"
    

print(counting)


# In[7]:


# Test of the enumeration loop
t.assertEqual(counting, "|0: mouse |1: rabbit |2: cat |3: dog |")


# ### String formating (0 P)
# 
# What does the following string formating result in?
# * Write the result of the string formating into the variables result1, result2, result3.
# * Example: `string0 = "This is a {} string.".format("test")`
# * Example solution: `result0 = "This is a test string"`

# In[8]:


# first string
string1 = "The sky is {}. {} words in front of {} random words create {} random sentence.".format(
    "clear", "Random", "other", 1
)

# second string
a = "irony"
b = "anyone"
c = "room"

string2 = f"The {a} of the situation wasn't lost on {b} in the {c}."

# third string
string3 = f"{7*10} * {9/3} with three digits after the floating point looks like this: {70*3 :.3f}."

# fourth string
string4 = "   Hello World.   ".strip()


# In[9]:


result1 = "The sky is clear. Random words in front of other random words create 1 random sentence."
result2 = "The irony of the situation wasn't lost on anyone in the room."
result3 = "70 * 3.0 with three digits after the floating point looks like this: 210.000."
result4 = "Hello World."


# In[10]:


# Test the string results
t.assertEqual(string1, result1)
t.assertEqual(string2, result2)
t.assertEqual(string3, result3)
t.assertEqual(string4, result4)


# # Homework 1: Python Basics
# 
# This first  exercise sheet tests the basic functionalities of the Python programming language in the context of a simple prediction task. We consider the problem of predicting health risk of subjects from personal data and habits. We first use for this task a decision tree.
# 
# ![](tree.png)
# 
# Make sure that you have downloaded the `tree.png` file from ISIS. For this exercise sheet, you are required to use only pure Python, and to not import any module, including `NumPy`.

# ## Classifying a single instance (15 P)
# 
# * In this sheet we will represent patient info as a tuple.
# * Implement the function `decision` that takes as input a tuple containing values for attributes (smoker,age,diet), and computes the output of the decision tree. Should return either `'less'` or `'more'`. No other outputs are valid.

# In[33]:


def decision(x: Tuple[str, int, str]) -> str:
    """
    This function implements the decision tree represented in the above image. As input the function
    receives a tuple with three values that represent some information about a patient.
    Args:
        x (Tuple[str, int, str]): Input tuple containing exactly three values.
        The first element represents a patient is a smoker this value will be 'yes'.
        All other values represent that the patient is not a smoker. The second
        element represents the age of a patient in years as an integer. The last
        element represents the diet of a patient. If a patient has a good diet
        this string will be 'good'. All other values represent that the patient
        has a poor diet.
    Returns:
        str: A string that has either the value 'more' or 
        'less'. No other return value is valid. ->>> preciso retornar uma string de more or less; 

    """
    dec = None
    
    if x[0] == "yes":
        if x[1] < 29.5:
            dec = "less"
        else:
            dec = "more"
    elif x[2] == "good":
        dec = "less"
    else:
        dec = "more"
    
    return dec


# In[34]:


# Test decision function

# Test expected 'more'
x = ("yes", 31, "good")
output = decision(x)
print(f"decision({x}) --> {output}")
t.assertIsInstance(output, str)
t.assertEqual(output, "more")

# Test expected 'less'
x = ("yes", 29, "poor")
output = decision(x)
print(f"decision({x}) --> {output}")
t.assertIsInstance(output, str)
t.assertEqual(output, "less")


# In[35]:


# This cell is for grading. DO NOT remove it


# ## Reading a dataset from a text file (10 P)
# In the previous task we created a method to classify the risk of patients, by manualy setting rules defining for which inputs the user is in `more` or `less` risk regarding their health. In the next exercises we will approach the task differently. Our goal is to create a classification method based on data. In order to achieve this we need to also create functions that loads the existing data into the program so that we can use it. Furthermore, we can use the loaded data to apply on our decision tree implementation and check what its outputs are.
# 
# The file `health-test.txt` contains several fictious records of personal data and habits. We split this task into two parts. In the first part, we assume that we have read a line from the file and can now process it. In the second function we load the file and process each line using the function we have defined for this purpose.
# 
# * Read the file automatically using the methods introduced during the lecture.
# * Represent the dataset as a list of tuples. Make sure that the tuples have the same format as in the previous task, e.g. `('yes', 31, 'good')`.
# * Make sure that you close the file after you have opened it and read its content. If you use a `with` statement then you don't have to worry about closing the file.
# 
# **Notes**: 
# * Make sure when opening a file not to use an absolute path. An absolute path will
# work on your computer, but when your code is tested on the departments computers it will fail. Use relative paths when opening files
# * Values read from files are always strings.
# * Each line contains a newline `\n` character at the end
# * If you are using Windows as your operating system, refrain from opening any text files using Notepad. It will remove any linebreaks `\n`. You should inspect the files using the Jupyter text editor or any other modern text editor.

# In[36]:


def parse_line_test(line: str) -> Tuple[str, int, str]:
    """
    Takes a line from the file, including a newline, and parses it into a patient tuple

    Args:
        line (str): A line from the `health-test.txt` file
    Returns:
        tuple: A tuple representing a patient
    """
    
    assert (
        line[-1] == "\n"
    ), "Did you change the contents of the line before calling this function?"
    entry = line.strip().split(",")
    entry[1] = int((entry[1]))  
    return tuple(entry)

    
    


# In[37]:


x = "yes,23,good\n"
parsed_line = parse_line_test(x)
smoker, age, diet = parsed_line
print(parsed_line)
t.assertIsInstance(parsed_line, tuple)
t.assertEqual(len(parsed_line), 3)
t.assertIsInstance(age, int)
t.assertNotIn("\n", diet, "Are you handling line breaks correctly?")
t.assertEqual(parsed_line[-1], "good")


# In[38]:


# This cell is for grading. DO NOT remove it


# In[39]:


def gettest() -> List[Tuple[str, int, str]]:
    """
    Opens the `health-test.txt` file and parses it
    into a list of patient tuples. You are encouraged to use
    the `parse_line_test` function but it is not necessary to do so.

    This functions assumes that the `health-test.txt` file is located in
    the same directory as this notebook.

    Returns:
        List[Tuple[str, int, str]]: A list of patient tuples as read
        from the file.
    """
    data = []
    with open('health-test.txt', 'r') as file:
        for line in file:
            smoker, age, diet = line.strip().split(',')
            data.append((smoker, int(age), diet))

    
    return data


# In[40]:


testset = gettest()
pprint(testset)
t.assertIsInstance(testset, list)
t.assertEqual(len(testset), 8)
t.assertIsInstance(testset[0], tuple)


# In[41]:


# This cell is for grading. DO NOT remove it


# ## Applying the decision tree to the dataset (15 P)
# 
# * Apply the decision tree to all points in the dataset, and return the ratio of them that are classified as "more".
# * A ratio is a value in [0-1]. So if out of 50 data points 15 return `"more"` the value that should be returned is `0.3`

# In[42]:


def evaluate_testset(dataset: List[Tuple[str, int, str]]) -> float:
    """
    Calculates the percentage of datapoints for which the
    decision function evaluates to `more` for a given dataset

    Args:
        dataset (List[Tuple[str, int, str]]): A list of patient tuples

    Returns:
        float: The percentage of data points which are evaluated to `'more'`
    """
    
    #sum([ sum(X**2) for x in data if x > 10])
    
    cnt = 0
    for x in dataset:
        dec = decision(x)
        if dec == "more":
            cnt += 1
    
    ratio = cnt / len(dataset)
    return ratio


# In[43]:


ratio = evaluate_testset(gettest())
print(f"ratio --> {ratio}")
t.assertIsInstance(ratio, float)
assert_percentage(ratio)
t.assertTrue(0.3 < ratio < 0.4)


# ## Learning from examples (10 P)
# 
# Suppose that instead of relying on a fixed decision tree, we would like to use a data-driven approach where data points are classified based on a set of training observations manually labeled by experts. Such labeled dataset is available in the file `health-train.txt`. The first three columns have the same meaning than for `health-test.txt`, and the last column corresponds to the labels.
# 
# * Read the `health-train.txt` file and convert it into a list of pairs. The first element of each pair is a triplet of attributes (the patient tuple), and the second element is the label.
# * Similarlly to the previous exercise we split the task into two parts. The first involves processing each line individually. The second handles opening the file and processing all lines of the file
# 
# **Note**: A triplet is a tuple that contains exactly three values, a pair is a tuple that contains exactly two values

# In[ ]:


def parse_line_train(line: str) -> Tuple[Tuple[str, int, str], str]:
    """
    This function works similarly to the `parse_line_test` function.
    It parses a line of the `health-train.txt` file into a tuple that
    contains a patient tuple and a label.

    Args:
        line (str): A line from the `health-train.txt`

    Returns:
        Tuple[Tuple[str, int, str], str]: A tuple that
        contains a patient tuple and a label as a string
    """
    assert line[-1] == "\n"
    entry = line.strip().split(',')
    entry[1] = int((entry[1])) 
    output = (tuple(entry[0:-1]), entry[-1]) 
    "duvida sobre os números [0:-1]"

    return output


# In[ ]:


x = "yes,67,poor,more\n"
parsed_line = parse_line_train(x)
print(parsed_line)

t.assertIsInstance(parsed_line, tuple)
t.assertEqual(len(parsed_line), 2)

data, label = parsed_line

t.assertIsInstance(data, tuple)
t.assertEqual(len(data), 3)
t.assertEqual(data[1], 67)

t.assertIsInstance(label, str)
t.assertNotIn("\n", label, "Are you handling line breaks correctly?")
t.assertEqual(label, "more")


# In[ ]:


# This cell is for grading. DO NOT remove it


# In[48]:


def gettrain() -> List[Tuple[Tuple[str, int, str], str]]:
    """
    Opens the `health-train.txt` file and parses it into
    a list of patient tuples accompanied by their respective label.

    Returns:
        List[Tuple[Tuple[str, int, str], str]]: A list
        of tuples comprised of a patient tuple and a label
    """
    data = []
    with open('health-train.txt', 'r') as file:
        for line in file:
            smoker, age, diet, label = line.strip().split(',')
            data.append( ( (smoker, int(age), diet), label) )
    
    return data


# In[49]:


trainset = gettrain()
pprint(trainset)
t.assertIsInstance(trainset, list)
t.assertEqual(len(trainset), 16)
first_datapoint = trainset[0]
t.assertIsInstance(first_datapoint, tuple)
t.assertIsInstance(first_datapoint[0], tuple)
t.assertIsInstance(first_datapoint[1], str)


# In[50]:


# This cell is for grading. DO NOT remove it


# In[ ]:





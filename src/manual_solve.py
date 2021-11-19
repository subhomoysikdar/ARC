#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

# Name: Subhomoy Sikdar
# ID: 21250101
# Github: https://github.com/subhomoysikdar/ARC

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

'''
In this task we have a colored sub grid which has random shape(s)
inside it. The output is to resize the grid to the colored sub
grid and then flip the shape(s) inside the colored grid.

Here we are using solve_1cf80156(x) function to remove the black
cells from the main grid. Then we are using the flipr method in 
numpy to flip the colored grid.

All the test and training grids are solved correctly.
'''


def solve_7468f01a(x):
    x = solve_1cf80156(x)
    x = np.fliplr(x)
    return x


'''
In this task we are given an abstract shape and we have to find
out the boundary of the shape. We need to then return the grid
with only the shape.

We are using numpy sum to project which rows and column have all
0's. Looking at the sum arrays we get the first and last index
where the value is not 0. This gives us the start and end row and
the start and end column. We then return the resized original array
based on the indices.

All the test and training grids are solved correctly.
'''


def solve_1cf80156(x):
    rows = np.sum(x, axis=1)
    cols = np.sum(x, axis=0)

    start_row = 0;
    end_row = len(rows)
    start_col = 0;
    end_col = len(cols)

    for i, v in enumerate(rows):
        if v != 0:
            start_row = i
            break

    for i, v in enumerate(reversed(rows)):
        if v != 0:
            end_row = len(rows) - i
            break

    for i, v in enumerate(cols):
        if v != 0:
            start_col = i
            break

    for i, v in enumerate(reversed(cols)):
        if v != 0:
            end_col = len(cols) - i
            break

    return x[start_row: end_row, start_col:end_col]


'''
In this task we are given a grid where there are random green cells.
If these green cells are connected to other green cells (not diaginally)
then mark them as blue. The output is a grid of same size where the
connected green cells are marked blue and isolated green cells remain green.

Here we are checking for every element if it has any adjoining elements
(not diagonal) and mark them in blue else leave them green. This is
done by the has_neighbours function where we check if the right/left/up/down
cells have a non-zero value. Boundaries are considered 0 i.e. the first cell
has 0 element as top and left. The code does not use any special library.

All the test and training grids are solved correctly.

P.S. The input to the function is correct but the input that gets printed is incorrect.
'''


def solve_67385a82(x):
    for i, row in enumerate(x):
        for j, item in enumerate(row):
            if x[i][j] == 3 and has_neighbours(x, i, j):
                x[i][j] = 8

    return x


def has_neighbours(x, i, j):
    rows = len(x)
    cols = (len(x[0]))

    left = False if j == 0 else x[i][j - 1] != 0
    right = False if j == cols - 1 else x[i][j + 1] != 0
    up = False if i == 0 else x[i - 1][j] != 0
    down = False if i == rows - 1 else x[i + 1][j] != 0

    return left or right or up or down


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

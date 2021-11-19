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
cells from the main grid. Then we are using the fliplr method in 
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


'''
In this task we are given a n * n grid with random colors. The final grid is 
a grid of size 2n * 2n which is composed of 4 original grids. The initial grid
is same but the right grid is the 90 degree clockwise rotation of the original
grid. Similarly, the bottom diagonal grid is 180 degree rotation and the bottom
grid is 270 degree rotation of the original grid. These 4 grids together forms
the final (n + n) * (n + n) grid.

We are using numpy rot90 function for rotating the original array create the 3
new grids. We are then joining the original array and 90 degree rotated grid using
numpy concatenate method. Similarly the below grids are concatenated horizontally.
The resultant grids are then concatenated vertically to produce the final output.

All the test and training grids are solved correctly.
'''


def solve_7fe24cdd(x):
    right = np.rot90(x, k=-1)
    diag = np.rot90(x, k=-2)
    down = np.rot90(x, k=-3)

    upper = np.concatenate((x, right), axis=1)
    lower = np.concatenate((down, diag), axis=1)

    x = np.concatenate((upper, lower), axis=0)

    return x


'''
In this task we are given a grid with some random cells marked blue. We have to check
if the blue cells can be joined with other blue cells horizontally or vertically.
Isolated cells will be left as is. The final output is a grid of same size where
the blue cells which have any left, right, up or down blue cells further along
the grid are joined by marking the intermediate cells as blue.

We figure out if there are blue cells which can be connected by looking at the
right elements and bottom elements. We start from the topmost corner and go over
each element. We are looking only → and ↓ directions but because we started from
the top corner we do not need to look in the up or left direction as those elements
would already been iterated. Also when we see an element in blue (cell value 8) we start
looking from the end of the array for any occurrnce of 8. This is because if we
have a scenario like [0, 0, 8, 0, 8, 0, 0, 8, 0] we don't stop at 5th element. Rather we 
look backwards so that we get the 8 occurring in 8th position (index 7). We
keep a list of these elements which needs to be joined by row or column in 2 lists
and join them after traversing the array. We cannot join them immediately because
if we update intermediate value to 8 that will impact when we scan later indices.

All the test and training grids are solved correctly.
'''


def solve_ded97339(x):
    row_joins = []
    col_joins = []

    for i, row in enumerate(x):
        for j, item in enumerate(row):
            if x[i][j] == 8:

                k = len(x[0]) - 1
                while x[i][k] == 0 and k > j:
                    k -= 1

                if k > j:
                    row_joins.append((i, j, k))  # ((i,j), (i,k))

                k = len(x) - 1
                while x[k][j] == 0 and k > i:
                    k -= 1

                if k > i:
                    col_joins.append((i, j, k))  # ((i,j), (k,j))

    for r in row_joins:
        for i in range(r[1], r[2] + 1):
            x[r[0]][i] = 8

    for c in col_joins:
        for i in range(c[0], c[2] + 1):
            x[i][c[1]] = 8

    return x


'''
In this task we have a red boundary inside the main grid. This red boundary
is continuous along 1 axis but open along the other axis. We have grey cells
inside this red rectangle and if the boundary is open along x axis then the 
grey cells are reflected horizontally. Here, the upper half of the space inside the
red rectangle is reflected horizontally on the top of the boundary and the 
lower half is reflected horizontally below the lower half of the boundary. Similary,
if the boundary is open vertically then the left half of the area inside the boundary
is reflected vertically on the left of the boundary and the right half reflected 
vertically outside the right boundary wall.

It is not necessary that the boundary will be placed in the center and the boundary
is a rectangle meaning the length and width of the boundary can be different. We first
check the grid for red cells (2) along the grid and then when we find the first cell
we traverse along the row and column to determine the length and width of the 
rectangle. We also note the axis (0 meaning rectangle open along x) and reflects
horizontally. Else axis is 1 and the reflection is along the vertical axis. We also
note the 4 corners of the rectangle as c1, c2, c3, and c4

c1 ----- c2
|        |
|        |
c3 ----- c4

From the corners and axis we find the mid point of the rectangle and reflect the left/right
for axis 0 and upper/lower for axis 0. We finally mark the internal grey cells as 0 and return
the grid. We have not used any special libraries for this code.

All the test and training grids are solved correctly.
'''


def solve_f8a8fe49(x):
    # get c1 - c4; tuples of x, y index of 4 corners of the red square
    # axis 0 - horizontal reflection, 1 - vertical reflection
    for i, row in enumerate(x):
        for j, item in enumerate(row):
            # reflects along horizontal line
            if item == 2 and row[j + 1] == 2 and row[j + 2] == 2:
                axis = 0
                c1 = (i, j)

                k = 2
                while row[j + k] == 2:  # check length of red squares traversing the row
                    k += 1
                c2 = (i, j + k - 1)

                k = 2
                while x[i + k][j] == 0:
                    k += 1
                c3 = (i + k + 1, j)

                c4 = (c3[0], c2[1])
                break

            # reflects along vertical line
            elif item == 2 and row[j + 1] == 2 and row[j + 2] == 0:
                axis = 1
                c1 = (i, j)

                k = 2
                while x[i + k][j] == 2:  # check length of red squares traversing the column
                    k += 1
                c3 = (i + k - 1, j)

                k = 2
                while row[j + k] == 0:
                    k += 1
                c2 = (i, j + k + 1)

                c4 = (c3[0], c2[1])
                break
            else:
                continue
        else:
            continue
        break

    # print(c1, c2, c3, c4)

    if axis == 0:
        mid = c1[0] + (c3[0] - c1[0]) // 2  # get row index for horizontal reflection

        # upper half
        for i in range(c1[0] + 1, mid + 1):
            for j in range(c1[1] + 1, c2[1]):
                if x[i][j] == 5:
                    x[c1[0] - (i - c1[0])][j] = 5  # mark reflection
                    x[i][j] = 0  # clear original

        # lower half
        for i in range(mid, c3[0]):
            for j in range(c1[1] + 1, c2[1]):
                if x[i][j] == 5:
                    x[c3[0] + (c3[0] - i)][j] = 5  # mark reflection
                    x[i][j] = 0  # clear original

    else:
        mid = c1[1] + (c2[1] - c1[1]) // 2  # get column index for vertical reflection

        # left half
        for i in range(c1[0] + 1, c3[0]):
            for j in range(c1[1] + 1, mid + 1):
                if x[i][j] == 5:
                    x[i][c1[1] - (j - c1[1])] = 5  # mark reflection
                    x[i][j] = 0  # clear original

        # right half
        for i in range(c1[0] + 1, c3[0]):
            for j in range(mid, c2[1]):
                if x[i][j] == 5:
                    x[i][c2[1] + (c2[1] - j)] = 5  # mark reflection
                    x[i][j] = 0  # clear original

    return x


'''
For some of the solutions above we have used numpy libraries whereas for some 
we have written with basic python functions. As in these functions we are
using 2D grids many numpy functions can be used to modify the array. For
example in the solve_f8a8fe49 function we have not used any libraries for 
reflecting the grey cells but we could have used numpy functions like flipud
and fliplr for up/down and left/right reflection. We could have split the numpy
array into two halves reflected based on the axis and concatenated back. This would 
generate the output with 4 red cells at incorrect position which we can fix because we
know the 4 corners of the rectangle.

Some other useful functions for these solutions are rot90 as many of these problems
involve rotation of the array. We can also use numpy where to match certain conditions
like in solve_1cf80156 after taking the sum along rows and column we can use np.where(cols != 0)
or np.where(rows != 0) to get a tuple which has the indices where the values are not 0.
We can also use numpy.diag to get diagonal elements and split and concatenate for splitting 
and concatenating the arrays.

Using these useful set of functions we can try to solve the solution for a given task. Here
the task is seen for the first time and the system has no prior knowledge or experience of
the task. Yet by using these functions (for example rotation and concatenate for solve_7fe24cdd)
we can correctly identify the output for all test inputs. The solve_1cf80156 which removes the 
surrounding 0's (or noise) to get the shape is another useful function. We have directly used this
in solve_7468f01a as 
- apply solve_1cf80156
- apply fliplr

Some other examples (not implemented above):
 
solve_963e52fc: fliplr + concatenate
solve_28bf18c6: solve_1cf80156 + concatenate (x + x)
solve_62c24649: fliplr + flipud
'''


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

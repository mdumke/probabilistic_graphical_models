## python-cheatsheet

materials collected for anaconda-python 3.5

## general

- to run a script from within a prompt, use `exec(open('my_script.py').read())`

## numpy

`np.zeros((2, 3, 4))` | create a three-dimensional numpy-array, filled with zeros
`my_3d_array.sum(axis = 1)` | sum a multi-dimensional np-array over the first axis
`sum(my_1d_array)` | an alternative way of doing summation
`'np.sum(my_matrix[:, 2])` | sum over all rows in the third column
`my_2d_array[0]` | pick the 0-entries along the first-dimension
`my_2d_array[:, 0]` | pick the 0-entries along the second-dimension
`np.outer(v1, v2)` | computes the outer-product-matrix of two vectors
`my_arr.shape` | returns the dimensions of an tensor
using numpy to import data from .dta-example
```
    movies = np.loadtxt(
      filename,
      dtype={
        'names': ('movieid', 'moviename'),
        'formats': ('int32', 'S100')},
      delimiter='\t')
```
`np.random.randint(1, 6 + 1, size = 10)` | generate an array of 10 random integers


## basic operations

`[some_func(x) for x in my_arr]` | apply a function to every element in my_arr



## plotting

- simple scatterplot:
```
import matplotlib.pyplot
import pylab
matplotlib.pyplot.scatter([1, 2], [1, 4])
matplotlib.pyplot.show()
```
- it is not necessary to provide y-values explicitely:
```
plt.figure()
plt.plot(x, np.log(x))
plt.plot(x, x - 1)
plt.xlabel('x')
plt.legend(['ln(x)', 'x - 1'], loc=4)
plt.show()
```





## sources

- computational probability and inference-course, edX
- stackoverflow, of course

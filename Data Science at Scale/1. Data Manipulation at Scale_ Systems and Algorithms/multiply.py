# Problem 6
# Assume you have two matrices A and B in a sparse matrix format, where each record is of the form i, j, value.
# Design a MapReduce algorithm to compute the matrix multiplication A x B
#
# Map Input
# The input to the map function will be a row of a matrix represented as a list. Each list will be of the form
# [matrix, i, j, value] where matrix is a string and i, j, and value are integers.
# The first item, matrix, is a string that identifies which matrix the record originates from. This field has two
# possible values: "a" indicates that the record is from matrix A and "b" indicates that the record is from matrix B.
#
# Reduce Output
# The output from the reduce function will also be a row of the result matrix represented as a tuple. Each tuple will be
# of the form (i, j, value) where each element is an integer.

import MapReduce
import sys

mr = MapReduce.MapReduce()


def mapper(record):
    if record[0] == 'a':
        for k in range(5):  # Here, here '5' refers to the number of columns in matrix B
            key = (record[1], k)
            value = ('a', record[2], record[3])
            mr.emit_intermediate(key, value)
    else:
        for i in range(5):  # Here, 5 refers to the number of rows in matrix A
            key = (i, record[2])
            value = ('b', record[1], record[3])
            mr.emit_intermediate(key, value)


def reducer(key, list_of_values):
    # key: (row, column) / (column, row)
    # values: (matrix identifier, column/row, value)
    hash_a = {}
    hash_b = {}

    for v in list_of_values:
        if v[0] == 'a':
            hash_a[v[1]] = v[2]
        else:
            hash_b[v[1]] = v[2]
    result = 0

    for j in range(5):
        try:
            result += hash_a[j] * hash_b[j]
        except:
            pass
    mr.emit(tuple(list(key) + [result]))


if __name__ == '__main__':
    inputdata = open(sys.argv[1])
    mr.execute(inputdata, mapper, reducer)

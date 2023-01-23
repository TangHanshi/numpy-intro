This tutorial covers the following topics:

- Working with numerical data in Python
- Going from Python lists to Numpy arrays
- Multi-dimensional Numpy arrays and their benefits
- Array operations, broadcasting, indexing, and slicing
- Working with CSV data files using Numpy

## Questions for Revision

Try answering the following questions to test your understanding of the topics covered in this notebook:

1. What is a vector? 
    A list of numbers.
2. How do you represent vectors using a Python list? Give an example.
    demo-vec = [1, 1, 3]
3. What is a dot product of two vectors? 
    element-wise multiplication of 2 vectors, and take the sum of the results
4. Write a function to compute the dot product of two vectors.
    def dot-product(v1, v2):
        result = 0
        for x, w in zip(v1, v2):
            result += x * w
        return result
5. What is Numpy?
    Numpy library provides Numerical built-in functions
6. How do you install Numpy?
    pip install Numpy
7. How do you import the `numpy` module?
    import numpy as np
8. What does it mean to import a module with an alias? Give an example.
    np
9. What is the commonly used alias for `numpy`?
    np
10. What is a Numpy array?
    Numpy provides convinient arrary manipulations
11. How do you create a Numpy array? Give an example.
    np.arrary()
12. What is the type of Numpy arrays?
    ndarray
13. How do you access the elements of a Numpy array?
    use indexing 
14. How do you compute the dot product of two vectors using Numpy?
    np.dot(v1, v2)
15. What happens if you try to compute the dot product of two vectors which have different sizes?
    ValueError
16. How do you compute the element-wise product of two Numpy arrays?
    np.matmul(v1, v2)
17. How do you compute the sum of all the elements in a Numpy array?
    np.sum
18. What are the benefits of using Numpy arrays over Python lists for operating on numerical data?
    Easy & Efficiency by C++  
19. Why do Numpy array operations have better performance compared to Python functions and loops?
    100 times faster 
20. Illustrate the performance difference between Numpy array operations and Python loops using an example.
    %%time
    result = 0
    for x1, x2 in zip(arr1, arr2):
        result += x1*x2
    result

    %%time
    np.dot(arr1_np, arr2_np)

21. What are multi-dimensional Numpy arrays? 
    nested arrays
22. Illustrate the creation of Numpy arrays with 2, 3, and 4 dimensions.
    2d = np.array([[1, 2], 
                   [1, 2]])
    3d = np.array([[[1, 2], [1, 2]], 
                   [[1, 2], [1, 2]]])      

23. How do you inspect the number of dimensions and the length along each dimension in a Numpy array?
    arr.shape
24. Can the elements of a Numpy array have different data types?
    no
25. How do you check the data type of the elements of a Numpy array?
    arr.dtype
26. What is the data type of a Numpy array?
27. What is the difference between a matrix and a 2D Numpy array?
28. How do you perform matrix multiplication using Numpy?
    np.matmul(m1, m2)
29. What is the `@` operator used for in Numpy?
    matrix multiplication 
    m1 @ m2
30. What is the CSV file format?
    comma seperated values
31. How do you read data from a CSV file using Numpy?
    climate_data = np.genfromtxt('climate.txt', delimiter=',', skip_header=1)
32. How do you concatenate two Numpy arrays?
    np.concatenate((arr1, arr2), axis=1)
33. What is the purpose of the `axis` argument of `np.concatenate`?
    The axis specifies the dimension for concatenation 
34. When are two Numpy arrays compatible for concatenation?
    The arrays should have the same number of dimensions, and the same length along each except the dimension used for concatenation. 
35. Give an example of two Numpy arrays that can be concatenated.
    arr1 = np.array([[1, 2], 
            [3, 4]])

    arr2 = np.array([[1, 2, 3], 
            [5, 5, 6]])

36. Give an example of two Numpy arrays that cannot be concatenated.
    arr1 = np.array([[1, 2], 
        [3, 4]])

    arr2 = np.array([1, 2, 3])

37. What is the purpose of the `np.reshape` function?
    format the array without changing data
38. What does it mean to “reshape” a Numpy array?
    format the array without changing data
39. How do you write a numpy array into a CSV file?
    np.savetxt('demo.txt', 
           arr, 
           fmt='%.2f', 
           delimiter=',',
           header='temperature,rainfall,humidity,yeild_apples', 
           comments='')

40. Give some examples of Numpy functions for performing mathematical operations.
    np.sum, np.exp, np.round
41. Give some examples of Numpy functions for performing array manipulation.
    np.array, np.reshape, np.stack, np.concatenate, np.split
42. Give some examples of Numpy functions for performing linear algebra.
    np.matmul, np.dot, np.transpose, np.eigvals
43. Give some examples of Numpy functions for performing statistical operations.
    np.mean, np.median, np.std, np.max 
44. How do you find the right Numpy function for a specific operation or use case?
    google 
45. Where can you see a list of all the Numpy array functions and operations?
    numpy.org
46. What are the arithmetic operators supported by Numpy arrays? Illustrate with examples.
    + - * / 
47. What is array broadcasting? How is it useful? Illustrate with an example.
    replication to 
48. Give some examples of arrays that are compatible for broadcasting?
49. Give some examples of arrays that are not compatible for broadcasting?
50. What are the comparison operators supported by Numpy arrays? Illustrate with examples.
    == 
    != 
    >= 
    <= 
    They compare every element within the arrays, result in booleans 
51. How do you access a specific subarray or slice from a Numpy array?
    # Subarray using ranges
    arr3[1:, 0:1, :2]
52. Illustrate array indexing and slicing in multi-dimensional Numpy arrays with some examples.
    # Subarray using ranges
    arr3[1:, 0:1, :2]
    # Mixing indices and ranges
    arr3[1:, 1, 3]
53. How do you create a Numpy array with a given shape containing all zeros?
    # All zeros
    np.zeros((3, 2))
54. How do you create a Numpy array with a given shape containing all ones?
    # All ones
    np.ones([2, 2, 3])
55. How do you create an identity matrix of a given shape?
    # Identity matrix
    np.eye(3)
56. How do you create a random vector of a given length?
    # Random vector
    np.random.rand(5)
57. How do you create a Numpy array with a given shape with a fixed value for each element?
    # Fixed value
    np.full([2, 3], 42)
58. How do you create a Numpy array with a given shape containing randomly initialized elements?
    np.random.randn(2, 3) 
59. What is the difference between `np.random.rand` and `np.random.randn`? Illustrate with examples.
    np.random.randn offers array shape, but np.random.rand does not. 
60. What is the difference between `np.arange` and `np.linspace`? Illustrate with examples.
    # Range with start, end and step
    np.arange(10, 90, 3)
    
    # Equally spaced numbers in a range
    np.linspace(3, 27, 9)

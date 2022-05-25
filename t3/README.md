## Task 3-1

Convert JPL to Hamilton: (x,y,z,w) -> (w,x,y,-z)

#### The output is :

```c++
transformation matrix from Camera Right to Camera Left is: 
    0.999903   -0.0139386 -0.000306394     0.102648
   0.0139382     0.999902  -0.00122039 -0.000909453
 0.000323375     0.001216     0.999999   0.00104141
           0            0            0            1
```



# Task 3-3

#### The output is :

```c++
The solution of normal equation is: m = 4, n = 1.49999
The solution of SVD decomposition is: m = 4, n = 1.49999
The solution of QR decomposition is: m = 4, n = 1.4999
```

reference: https://www.codeleading.com/article/68632016892/





# Task 3-4

```

```

The condition number is:
$$
k(A) = ||A^{-1}||||A||
$$
Here we use L2 norm, In L2 norm, comdition number is max singular value/ min singular value.
一个低条件数的问题称为良态的，而高条件数的问题称为病态（或者说非良态）的.

The output is:

```
The solution of normal equation is: m = 4.58963, n = -0.811457
The solution of SVD decomposition is: m = 4.58963, n = -0.811457
The solution of QR decomposition is: m = 4.58963, n = -0.811457
Single Values of Data 1: 58.5265
4.98196
Single Values of Data 2: 41.0307
1.00646
Condition number of Data 1: 11.7477
Condition number of Data 2: 40.7673

```


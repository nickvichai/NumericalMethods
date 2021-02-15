from pprint import pprint
import scipy
import scipy.linalg
import math
import matplotlib.pyplot as plt

def generateMatrixA(m):
    n = m * m 
    A = [[0 for x in range(n)]
            for y in range(n)]
    
    for i in range(n):
        A[i][i] = 4
    for i in range(n-1):
        A[i+1][i] = -1
        A[i][i+1] = -1
    
    return scipy.array(A)
     

def generateMatrixD(m, A):
    n = m * m
    D = [[0 for x in range(n)]
            for y in range(n)]
    
    for i in range(n):
        D[i][i] = math.sqrt(A[i][i])

    return scipy.array(D)

def generateMatrixI(m):
    n = m * m
    I = [[0 for x in range(n)]
            for y in range(n)]
    
    for i in range(n):
        I[i][i] = 1

    return scipy.array(I)

def createLogGraph(m_vals, e1_vals, e2_vals, e3_vals):
    plt.loglog(m_vals, e1_vals)
    plt.loglog(m_vals, e2_vals)
    plt.loglog(m_vals, e3_vals)
    plt.show()
    return

def compute(m):
    #generate matrix A
    A = generateMatrixA(m)

    """
    print("Matrix A")
    pprint(A)
    print("\n")
    """
    
    #part (a) -- generate LU decomp
    P, L, U = scipy.linalg.lu(A)

    """
    print("Matrix L")
    pprint(L)
    print("Matrix U")
    pprint(U)
    print("\n")
    """

    #part (b) -- compute Cholesky decomp
    D = generateMatrixD(m, A)

    L_tilda = L.dot(D)
    L_tilda_t = L_tilda.T

    """
    print("Matrix L~")
    pprint(L_tilda)
    print("Matrix L~ Transposed")
    pprint(L_tilda_t)
    

    print("Matrix A from Cholesky Decomp")
    pprint(L_tilda.dot(L_tilda_t))
    print("\n")
    """

    #part (c) -- compute inverse A
    L_inv = scipy.linalg.inv(L)
    U_inv = scipy.linalg.inv(U)

    """
    print("Matrix L^-1")
    pprint(L_inv)
    print("Matrix U^-1")
    pprint(U_inv)
    

    print("Matrix A^-1 from LU decmop")
    pprint(L_inv.dot(U_inv))
    print("\n")
    """

    #part (d) -- compute norms and add to graph
    e1 = scipy.linalg.norm(A - (L.dot(U)))
    e2 = scipy.linalg.norm(A - (L_tilda.dot(L_tilda_t)))
    I = generateMatrixI(m)
    e3 = scipy.linalg.norm(A.dot(L_inv.dot(U_inv)) - I)

    return [e1, e2, e3]



if __name__ == "__main__":
    m_vals = [5, 10, 50, 100, 500]
    error_norms = [[] for x in range(3)]
    for m in m_vals: 
        print("Computing m value: " + str(m))
        errors = compute(m)
        error_norms[0].append(errors[0])
        error_norms[1].append(errors[1])
        error_norms[2].append(errors[2])

    createLogGraph(m_vals, error_norms[0], error_norms[1], error_norms[2])

    

   





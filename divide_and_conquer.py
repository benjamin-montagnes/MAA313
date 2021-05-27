import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt
import matplotlib.tri as tri
import matplotlib.collections
from random import randrange

#_______________________________________________________________________________

# We define the different functions needed for the Triangulation

#Clockwise (CW) and Counter Clockwise (CCW)
CW = 1
CCW = -1

def QTEST(H, I, J, K):
    """
    QTEST(H,I,J,K) tests the quadrilateral having CCW ordered vertices H,I,J,K. 
    It returns TRUE if the circumcircle of AHIJ does not contain K in its 
    interior, and returns FALSE otherwise
    """
    mat= [[H[0]-K[0],H[1]-K[1],(H[0]-K[0])**2+(H[1]-K[1])**2],
          [I[0]-K[0],I[1]-K[1],(I[0]-K[0])**2+(I[1]-K[1])**2],
          [J[0]-K[0],J[1]-K[1],(J[0]-K[0])**2+(J[1]-K[1])**2]]
    det = np.linalg.det(mat)
    if det > 0: return True
    return False

def orientation(a, b, c):
    det=np.linalg.det(np.array(([[b[0]-a[0],b[1]-a[1]],[c[0]-a[0],c[1]-a[1]]])))
    if det > 0: return CW
    elif det == 0: return 0
    return CCW

def split(l, key=None, k=7, limmed=100):
  def m1(l, key=None):
    n = len(l)
    l = sorted(l, key=key)
    return l[n//2]
    
  def m2(l, s, e, key=None, k=7, limmed=100):
    if e - s <= limmed: return m1(l[s:e], key)
    recurs = [m2(l,s+(i*(e-s))//k,s+((i+1)*(e-s))//k,key,k,limmed) for i in range(k)]
    return m1(recurs, key)
  
  return m2(l, 0, len(l), key, k, limmed)

def inv(p):
    return (p[1], p[0])

def VAR(points):
    n = len(points)
    sum_x,sum_y,sum_sqx,sum_sqy = 0,0,0,0
    for (x,y) in points:
        sum_x += x
        sum_y += y
        sum_sqx += x**2
        sum_sqy += y**2
    var_x = (sum_sqx/n)-(sum_x/n)**2
    var_y = (sum_sqy/n)-(sum_y/n)**2
    return var_x, var_y
  
#_______________________________________________________________________________

# We de the Delaunay Triangulation using divide and conquer

def delaunay_triangulation(V):
    """
    PRED(vi, vj) denotes the point vi which appears clockwise (CW) of and 
    immediately after the point vij. The counter-clockwise function SUCC operates
    in a similar manner.
    """
    SUCC, PRED, FIRST = {}, {}, {}

    def DELETE(A, B):
        """
        DELETE(A, B) deletes A from the adjacency list of B and B from the 
        adjacency list of A
        """
        SA, SB = SUCC.pop((A, B)), SUCC.pop((B, A))
        PA, PB = PRED.pop((A, B)), PRED.pop((B, A))
        SUCC[A, PA], SUCC[B, PB] = SA, SB
        PRED[A, SA], PRED[B, SB] = PA, PB

    def INSERT(A, B, SA, PB):
        """
        INSERT(A, B) inserts point A into the adjacency list of B and point B 
        into the adjacency list of A at proper positions
        """
        PA, SB = PRED[A, SA], SUCC[B, PB]
        SUCC[B, A], SUCC[A, B] = SB, SA
        SUCC[A, PA], SUCC[B, PB] = B, A
        PRED[A, SA], PRED[B, SB] = B, A
        PRED[B, A], PRED[A, B] = PB, PA
        
    def HULL(H1, H2):
        """
        HULL is input two convex hulls. It finds their lower common tangent
        """
        X, Y = H1, H2
        Z, Z1 = FIRST[Y], FIRST[X]
        Z2 = PRED[X, Z1]
        while True:
            if orientation(X, Y, Z) == CCW: Y, Z = Z, SUCC[Z, Y]
            elif orientation(X, Y, Z2) == CCW: X, Z2 = Z2, PRED[Z2, X]
            else: return (X, Y)

    def MERGE(X, Y):
        """
        MERGE is input two triangulations and the upper and lower common tangents
        of their convex hulls. It merges the two triangulations, starting with 
        the lower common tangent, zigzagging upward until the upper common 
        tangent is reached.
        """
        INSERT(X, Y, FIRST[X], PRED[Y, FIRST[Y]])
        FIRST[X] = Y
        while True: 
            if orientation(X, Y, PRED[Y, X]) == CW:
                Y1 = PRED[Y, X]
                Y2 = PRED[Y, Y1]
                while QTEST(X, Y, Y1, Y2):
                    DELETE(Y, Y1)
                    Y1 = Y2
                    Y2 = PRED[Y, Y1]
            else:
                Y1 = None
            if orientation(X, Y, SUCC[X, Y]) == CW:
                X1 = SUCC[X, Y]
                X2 = SUCC[X, X1]
                while QTEST(X, Y, X1, X2):
                    DELETE(X, X1)
                    X1 = X2
                    X2 = SUCC[X, X1]
            else:
                X1 = None
            if X1 is None and Y1 is None:
                break
            elif X1 is None:
                INSERT(Y1, X, Y, Y)
                Y = Y1
            elif Y1 is None:
                INSERT(Y, X1, X, X)
                X = X1
            elif QTEST(X, Y, Y1, X1):
                INSERT(Y, X1, X, X)
                X = X1
            else:
                INSERT(Y1, X, Y, Y)
                Y = Y1
        FIRST[Y] = X

    def DT(V):
        n = len(V)
        if n == 2:
            [a, b] = V
            SUCC[a, b] = PRED[a, b] = b
            SUCC[b, a] = PRED[b, a] = a
            FIRST[a], FIRST[b] = b, a
        elif n == 3:
            [a, b, c] = V
            if orientation(a, b, c) == CW:
                SUCC[a, c] = SUCC[c, a] = PRED[a, c] = PRED[c, a] = b
                SUCC[a, b] = SUCC[b, a] = PRED[a, b] = PRED[b, a] = c
                SUCC[b, c] = SUCC[c, b] = PRED[b, c] = PRED[c, b] = a
                FIRST[a], FIRST[b], FIRST[c] = b, c, a
            elif orientation(a, b, c) == CCW:
                SUCC[a, b] = SUCC[b, a] = PRED[a, b] = PRED[b, a] = c
                SUCC[a, c] = SUCC[c, a] = PRED[a, c] = PRED[c, a] = b
                SUCC[b, c] = SUCC[c, b] = PRED[b, c] = PRED[c, b] = a
                FIRST[a], FIRST[b], FIRST[c] = c, a, b
            else:
                [a, b, c] = sorted(V)
                SUCC[a, b] = PRED[a, b] = SUCC[c, b] = PRED[c, b] = b
                SUCC[b, a] = PRED[b, a] = c
                SUCC[b, c] = PRED[b, c] = a
                FIRST[a], FIRST[c] = b, b
                
        else:
            varx, vary = VAR(V)
            if vary < varx:
                med = split(V)
                V_L = [p for p in V if p < med]
                V_R = [p for p in V if p >= med]
                DT(V_L)
                DT(V_R)
                X, Y = HULL(max(V_L), min(V_R))
                MERGE(X, Y)
            else:
                med = split(V, key=inv)
                V_D = [p for p in V if inv(p) < inv(med)]
                V_U = [p for p in V if inv(p) >= inv(med)]
                DT(V_D)
                DT(V_U)
                X, Y = HULL(max(V_D, key=inv), min(V_U, key=inv))
                MERGE(X, Y)
            
    DT(V)
    return SUCC
  
#_______________________________________________________________________________

# Test of our algorithm with random points

def intpoints(n, nbpoints):
    points = set()
    x, y = randrange(nbpoints), randrange(nbpoints)
    for i in range(n):
        while (x, y) in points: x, y = randrange(nbpoints), randrange(nbpoints)
        points.add((x, y))
    return list(points)

radius = 100
seeds = intpoints(100,100)
dt = delaunay_triangulation(seeds)

#fig, ax = plt.subplots()
fig = plt.figure(figsize=(15,15))
ax.margins(0.1)
ax.set_aspect('equal')
plt.axis([-1, radius+1, -1, radius+1])
xval, yval = [],[]
for i in dt:
  plt.plot([list(i[0])[0],list(i[1])[0]],[list(i[0])[1],list(i[1])[1]], 'steelblue', zorder=1)
for i in seeds:
  plt.scatter(i[0], i[1], c='green', zorder=2)
plt.show()

#_______________________________________________________________________________

# The result we should obtain

from scipy.spatial import Delaunay
seeds2 = np.array([list(i) for i in seeds])
fig = plt.figure(figsize=(15,15))
tri = Delaunay(seeds2)
plt.triplot(seeds2[:,0], seeds2[:,1], tri.simplices)
plt.plot(seeds2[:,0], seeds2[:,1], 'o')
plt.show()

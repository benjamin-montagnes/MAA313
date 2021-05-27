import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from random import randrange

#_______________________________________________________________________________

#Define the different Classes
def ccw(a,b,c): # is counter clockwise
  a,b,c = np.array([a,b,c])
  return np.cross(b-a, c-a) > 0

class Point :
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def  __eq__(self, other) :
    if (self.x == other.x) and (self.y == other.y): return True
    return False
  
  def print(self):
    return (self.x,self.y)

  def IsInCircumcircleOf(self, T): #T type triangle
    
    *tri,p = [(p.x, p.y) for p in [T.vx1, T.vx2, T.vx3, self]]
    if not ccw(*tri):
      tri[1],tri[2] = tri[2],tri[1]

    tri = np.array(tri) - p
    M = np.c_[tri, np.einsum('ij,ij->i', tri, tri).T]
    return np.linalg.det(M) > 0

class Edge :
  def __init__(self, p1, p2) :
    self.p1 = p1
    self.p2 = p2
  
  def  __eq__(self, other) :
    return (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1) 

class Triangle :
  def __init__(self, vx1, vx2, vx3):
    self.vx1 = vx1
    self.vx2 = vx2
    self.vx3 = vx3
  
  def  __eq__(self, other) :
    if (self.vx1 == other.vx1 or self.vx1 == other.vx2 or self.vx1 == other.vx3) \
    and (self.vx2 == other.vx1 or self.vx2 == other.vx2 or self.vx2 == other.vx3) \
    and (self.vx3 == other.vx1 or self.vx3 == other.vx2 or self.vx3 == other.vx3) :
      return True 
    return False 

  def edges(self):
    return [Edge(self.vx1, self.vx2), Edge(self.vx2,self.vx3), Edge(self.vx1,self.vx3)]

  def point_not_in_edge(self, Edge) :
      for point in [self.vx1, self.vx2, self.vx3] :
        if point != Edge.p1 and point != Edge.p2 :
          return point    

  def neighbor(self,other):
    for edge1 in self.edges() :
      for edge2 in other.edges() :
        if edge1 == edge2 : return True 
    return False

class Rectangle:
  def __init__(self, vx1, vx2, vx3, vx4) : #type Point
    self.vx1 = vx1 #top left  
    self.vx2 = vx2 #top right
    self.vx3 = vx3 #bottom left
    self.vx4 = vx4 #bottom right

  def length(self) :
    return abs(self.vx2.x - self.vx1.x)
  
  def breadth(self) :
    return abs(self.vx1.y - self.vx3.y)

  def list_vertices(self) :
    return [self.vx1,self.vx2,self.vx3,self.vx4]

  def partition(self, number) :
      bins = []
      nx = int(math.sqrt(number)) #number of columns
      ny = nx #number of rows
      #Partition the rectangle into number smaller rectangular regions (j'ai fait diviser verticallement)
      new_rec_length = self.length() / nx
      new_rec_breadth = self.breadth() / ny
      for i in range(ny) :
        for j in range(nx) :
          if (i%2==1) :
            j = nx - j - 1 #we go to the left direction on odd numbered lines
          top_left = Point(self.vx1.x + j * new_rec_length, self.vx1.y - i * new_rec_breadth)
          top_right = Point(self.vx1.x + (j+1) * new_rec_length, self.vx1.y - i * new_rec_breadth)
          bottom_left = Point(self.vx1.x + j * new_rec_length, self.vx1.y - (i+1) * new_rec_breadth)
          bottom_right = Point(self.vx1.x + (j+1) * new_rec_length, self.vx1.y - (i+1) * new_rec_breadth)
          bins.append(Rectangle(top_left,top_right,bottom_left,bottom_right))
      return bins

  def is_point_in_rectangle(self, p) : #p type Point
    return (p.x >= self.vx1.x) and (p.x <= self.vx2.x) \
      and (p.y >= self.vx3.y) and (p.y <= self.vx1.y)


def PointInTriangle (pt, T): #pt type point, T type Triangle
  def sign(p1, p2, p3) : #p1,p2,p3 type Point
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);

  d1 = sign(pt, T.vx1, T.vx2);
  d2 = sign(pt, T.vx2, T.vx3);
  d3 = sign(pt, T.vx3, T.vx1);

  has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0);
  has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0);

  return not (has_neg and has_pos);

def show_tris(ts, ax=None):
  if not ax:
    fig, ax = plt.subplots(figsize=(5,5))
  pts = np.array([(p.x,p.y) for t in ts for p in (t.vx1, t.vx2, t.vx3)])
  x,y = pts.T
  tri = np.arange(len(pts)).reshape(-1,3)
  ax.triplot(x,y,tri)

def should_swap(T1,T2) :
  for edge in T1.edges() :
    for other_edge in T2.edges() :
      if edge == other_edge :
        p4 = T2.point_not_in_edge(other_edge)
        if p4.IsInCircumcircleOf(T1) : return True
  return False

def swap(T1,T2) :
  for edge in T1.edges() :
    for other_edge in T2.edges() :
      if edge == other_edge :
        p4 = T2.point_not_in_edge(other_edge)
        p1 = T1.point_not_in_edge(edge)
        p2, p3 = edge.p1, edge.p2
        x1 = Triangle(p1,p2,p4)
        x2 = Triangle(p1,p3,p4)
        return x1,x2
  raise ValueError('no edge in common')

#_______________________________________________________________________________

# Geometry tests

fig, ax = plt.subplots(2,2, figsize=(15,15))
pts = np.random.rand(4000,2)
Pts = [Point(x,y) for (x,y) in pts]

vtx = [(.3,.3), (.8,.3), (.5,.8)]
t1 = Triangle(*[Point(x,y) for (x,y) in vtx])
t2 = Triangle(*[Point(x,y) for (x,y) in vtx[::-1]])

inside = [PointInTriangle(p,t1) for p in Pts]
ax[0,0].scatter(*pts[inside].T)

inside = [PointInTriangle(p,t2) for p in Pts]
ax[0,1].scatter(*pts[inside].T)

inside = [p.IsInCircumcircleOf(t1) for p in Pts]
ax[1,0].scatter(*pts[inside].T)

inside = [p.IsInCircumcircleOf(t2) for p in Pts]
ax[1,1].scatter(*pts[inside].T);
  
#_______________________________________________________________________________

# Swap tests

a,b,c,d = [[0,0],[0.3,.7],[1,1],[1,0]]
t1 = Triangle(Point(*a),Point(*b),Point(*c))
t2 = Triangle(Point(*a),Point(*c),Point(*d))

x1, x2 = swap(t1,t2)
print(should_swap(t1,t2)) # == True
print(should_swap(x1,x2)) # == False

fig, ax = plt.subplots(1,2,figsize=(10,5))
show_tris([t1,t2],ax[0])
show_tris([x1,x2],ax[1])

#_______________________________________________________________________________

# inside circle

fig, ax = plt.subplots(figsize=(5,5))
tri = Point(0, .2), Point(0.2, 0), Point(.3, 0)
T = Triangle(*tri)
pts = np.random.rand(1000, 2)
inside = np.array([Point(x,y).IsInCircumcircleOf(T) for x,y in pts])
plt.scatter(*pts[~inside].T)

#_______________________________________________________________________________

# inside triangle

fig, ax = plt.subplots(figsize=(5,5))
tri = Point(0, .2), Point(1, 0), Point(1, 1)
T = Triangle(*tri)
pts = np.random.rand(1000, 2)
inside = np.array([PointInTriangle(Point(x,y),T) for x,y in pts])
plt.scatter(*pts[~inside].T)

#_______________________________________________________________________________

# Tribuild Algorithm

def Swap_triangle(Triangulation):
    for T in Triangulation :
        for other in Triangulation :
          if T.neighbor(other) :
            if should_swap(T, other) :
              a, b = swap(T,other) #Swap a diagonal with its alternate
              Triangulation.remove(T) 
              Triangulation.remove(other) 
              Triangulation.append(a)
              Triangulation.append(b)
              return True
    return False

def Tribuild(V, R, reorder=True) : #V type list, R type Rectangle
  #beginning of INITIALIZATION
  N = len(V)
  vertices_rec = R.list_vertices()
  #remove any points which fall on the vertices of the rectangle
  for point in V :
    for vertex in vertices_rec :
      if point == vertex :
        V.remove(point)
  #Partition the rectangle into approximately N1/2 bins (smaller rectangular regions)
  bins = R.partition(N**(1/2)) 
  #Reorder the points by bins, starting at some bin and proceeding to neighboring bins
  points_reordered = []
  for bin in bins :
    for point in V :
      if bin.is_point_in_rectangle(point) : 
        points_reordered.append(point)

  if not reorder:
    points_reordered = V
  
  #Place the first point into the rectangle.
  first_point = points_reordered[0]

  #Connect the point to the four corners of the rectangle to produce an initial triangulation
  Triangulation = []
  Triangle_1 = Triangle(first_point, vertices_rec[0], vertices_rec[1])
  Triangle_2 = Triangle(first_point, vertices_rec[0], vertices_rec[2])
  Triangle_3 = Triangle(first_point, vertices_rec[1], vertices_rec[3])
  Triangle_4 = Triangle(first_point, vertices_rec[2], vertices_rec[3])
  Triangulation.append(Triangle_1) 
  Triangulation.append(Triangle_2)
  Triangulation.append(Triangle_3)
  Triangulation.append(Triangle_4)

  #end of INITIALIZATION
  #beginning of ITERATION 

  #If all points in V have been used then stop
  #step 1 : Input the next point to the existing triangulation. Connect this point to the vertices of its enclosing triangle.
  for i in range(1,len(V)) :
    point = points_reordered[i]
    for triangle in Triangulation :
      if PointInTriangle(point, triangle) :
          Triangulation.remove(triangle)
          Triangle_1 = Triangle(point, triangle.vx1, triangle.vx2)
          Triangle_2 = Triangle(point, triangle.vx2,triangle.vx3)
          Triangle_3 = Triangle(point,triangle.vx1,triangle.vx3)
          for T in [Triangle_1, Triangle_2, Triangle_3] :
            Triangulation.append(T)
          break #we inserted the new triangle
    for _it in range(500):
      if not Swap_triangle(Triangulation):
        break
    else:
      raise ValueError('too many swaps')
  return Triangulation     

#_______________________________________________________________________________

# Test of our algorithm with precise points

rec = Rectangle(Point(0,1), Point(1,1), Point(0,0),Point(1,0))
points = [(0.1,0.5), (0.2,0.6), (0.5,0.4), (0.2,0.9)]
V = [Point(x,y) for (x,y) in points]
Triangulation = Tribuild(V, rec)
show_tris(Triangulation)

#_______________________________________________________________________________

# The result we should obtain

x = [p.x for p in V] + [p.x for p in rec.list_vertices()]
y = [p.y for p in V] + [p.y for p in rec.list_vertices()]
fig, ax = plt.subplots(figsize=(10,10))
ax.triplot(x,y);

#_______________________________________________________________________________

# Testing with random points

def intpoints(n, nbpoints):
    points = set()
    x, y = randrange(nbpoints), randrange(nbpoints)
    for i in range(n):
        while (x, y) in points: x, y = randrange(nbpoints), randrange(nbpoints)
        points.add((x, y))
    return list(points)

rec = Rectangle(Point(0,100), Point(100,100), Point(0,0),Point(100,0))
points = genere(5,100)
print(points)
V = [Point(x,y) for (x,y) in points]
Triangulation = Tribuild(V, rec)
show_tris(Triangulation)

#_______________________________________________________________________________

# The result we should obtain

from scipy.spatial import Delaunay
seeds2 = np.array([list(i) for i in points])
tri = Delaunay(seeds2)
plt.triplot(seeds2[:,0], seeds2[:,1], tri.simplices)
plt.plot(seeds2[:,0], seeds2[:,1], 'o')
plt.show()

  

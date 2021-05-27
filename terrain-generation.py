import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

#_______________________________________________________________________________

# Define the necessary functions

def landscape(x,y,simga_x,simga_y,mu_x,mu_y,M, height):
    return (-1)**M * np.exp(-simga_x*(x-mu_x)**2 - simga_x*(y-mu_y)**2) * height

class Terrain:
    def __init__(self, nb_centre,xmin,xmax,ymin,ymax):
        #xmin,xmax,ymin,ymax = dimension of the grid
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.nb_centre = nb_centre #the number of mountains and valleys [mountain, valleys]

        self.centre = [[random.randint(xmin,xmax),random.randint(ymin,ymax)] for _ in range(self.nb_centre[0] + self.nb_centre[1]) ] #array of two rows which deﬁnes the coordinate of each mountain aka center/mu

        self.etal = [[0.001*random.randint(1,10), 0.001*random.randint(1,10)] for _ in range(self.nb_centre[0] + self.nb_centre[1])] #array of two rows which deﬁnes how spread out each mountain/valley should be aka sigma

        self.height = [random.random()*((xmax-xmin)) for _ in range(self.nb_centre[0] + self.nb_centre[1])]  #vector which deﬁnes the height of each mountain [h1, h2, ...] + valleys [-h3, -h4, ...]

        self.mountain = [2]*self.nb_centre[0] + [1]*self.nb_centre[1] #vector formed by booleans 2 = mountain / 1 = valley

        
def show_dataset(terrain, x,y,z):
    print(terrain.nb_centre)
    print(terrain.centre)
    print(terrain.etal)
    print(terrain.height)
    print(terrain.mountain)

    fig = plt.figure(figsize=(25,25))
    indices_1 = [331,332,333]

    for i in range(3):
      ax = fig.add_subplot(indices_1[i], projection='3d')
      ax.scatter(x,y,z)
      ax.view_init(azim=i*90, elev=20)
      ax.set(xlim=(terrain.xmin, terrain.xmax), ylim=(terrain.ymin, terrain.ymax), zlim=(-terrain.xmax, terrain.xmax))
      ax.set_xlabel('X Label')
      ax.set_ylabel('Y Label')
      ax.set_zlabel('Z Label')

    plt.show()


def moutain_construct_old(terrain): #construct the terrain
    result = [[0]*terrain.ymax]*terrain.xmax
    for x in range(terrain.xmax):
        for y in range(terrain.ymax):
          result[x][y] = landscape(x,y,terrain.etal[0][0],terrain.etal[0][1],terrain.centre[0][0],terrain.centre[0][1],terrain.mountain[0], terrain.height[0])
          if abs(result[x][y]) > 2: print(result[x][y])
    
    return result
  

def moutain_construct(terrain, nbr=100): #construct the terrain
    x_list, y_list, z_list = [], [], []
    for _ in range(nbr):
      x = random.random() * (terrain.xmax - terrain.xmin)
      y = random.random() * (terrain.ymax - terrain.ymin)
      z = 0
      for i in range(terrain.nb_centre[0]+terrain.nb_centre[1]):
          temp = landscape(x,y,terrain.etal[i][0],terrain.etal[i][1],terrain.centre[i][0],terrain.centre[i][1],terrain.mountain[i], terrain.height[i]) 
          if (abs(temp) > abs(z)):
            z = temp 
      #if abs(z) > 1: 
      x_list.append(x)
      y_list.append(y)
      z_list.append(z)
      
    return x_list, y_list, z_list


def squeeze_terrain(x,y,nbr=100):
    return [ (x[i],y[i]) for i in range(nbr) ]
  
#_______________________________________________________________________________

# Build the terrain

terrain = Terrain([1, 2],0,100,0,100)
x,y,z = moutain_construct(terrain)
show_dataset(terrain, x,y,z)  

#_______________________________________________________________________________



points = squeeze_terrain(x,y)
rec = Rectangle(Point(0,100), Point(100,100), Point(0,0),Point(100,0))
V = [Point(x,y) for (x,y) in points]
print(points)
print(V)
Triangulation = Tribuild(V, rec)
show_tris(Triangulation)

#_______________________________________________________________________________


tri = Delaunay(points)
dt = delaunay_triangulation(points)
fig = plt.figure(figsize=(15,15))
plt.triplot(np.array(points)[:,0], np.array(points)[:,1], tri.simplices)
plt.plot(np.array(points)[:,0], np.array(points)[:,1], 'o')
plt.show()










  

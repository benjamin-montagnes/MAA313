import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from divide-and-conquer import delaunay_triangulation
from tribuild import Tribuild, Point, Rectangle

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


def moutain_construct_uniform(terrain,nbr=15): #construct the terrain
    result = [[0]*terrain.ymax]*terrain.xmax
    x_list, y_list, z_list = [], [], []
    step = terrain.xmax / nbr 
    for j in range(nbr):
        x = j * step
        for k in range(nbr):
            y = k * step 
            z = 0
            for i in range(terrain.nb_centre[0]+terrain.nb_centre[1]):
                temp = landscape(x,y,terrain.etal[i][0],terrain.etal[i][1],terrain.centre[i][0],terrain.centre[i][1],terrain.mountain[i], terrain.height[i]) 
                if (abs(temp) > abs(z)):
                  z = temp 
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
    
    return x_list, y_list, z_list 
  

def moutain_construct_random(terrain, nbr=100): #construct the terrain
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


def squeeze_terrain(x,y):
    return [ (x[i],y[i]) for i in range(len(x)) ]
  
#_______________________________________________________________________________

# Build the terrain randomly sampled

terrain = Terrain([1, 2],0,100,0,100)
x,y,z = moutain_construct_random(terrain)
show_dataset(terrain, x,y,z)  

# Build the terrain uniformly sampled
# x,y,z = moutain_construct_uniform(terrain)

#_______________________________________________________________________________

#Triangulation using Tribuild

points = squeeze_terrain(x,y)
dt = delaunay_triangulation(points)

#_______________________________________________________________________________

#2D Result

radius = 100
fig, ax = plt.subplots(figsize=(20,20))
ax.margins(0.1)
ax.set_aspect('equal')
plt.axis([-1, radius+1, -1, radius+1])
for i in points:
  plt.scatter(i[0], i[1])
xval, yval = [],[]
for i in dt:
  plt.plot([list(i[0])[0],list(i[1])[0]],[list(i[0])[1],list(i[1])[1]], 'steelblue', zorder=1)
triangles = [[i[0],i[1],dt[i]] for i in dt]
t_num=[(cx.index(i[0][0]),cx.index(i[1][0]),cx.index(i[2][0])) for i in triangles]
plt.show()







  

from itertools import product
import math
import  random

def smoothstep(t):
    # makes the noise smother than if we only used the linear interpolation
    return t*t*t*(6*t**2 - 15*t + 10)

def lerp(a, b, t):
    # linear interpolation between two points
    return a + (b - a) * t

class PerlinNoise(object):

    def __init__(self, dimension, octaves=1, tile=(), unbias=False):
        self.dimension = dimension
        self.tile = tile+(0,)*dimension
        self.octaves = octaves
        self.unbias = unbias
        self.scale_factor= 2 * dimension**-0.5
        self.gradient={}

    def generate_gradient(self):

        #in case of one dimension the noise is trivial, we use a slope between [-1,1]
        if self.dimension == 1:
            return (random.uniform(-1,1),)

        #in other cases we need to get random unit vectors, for this what we do is pick a random point in the surface of an n-hypersphere (which is equivalent)
        #first we get random points with std of 1
        random_p=[random.gauss(0,1) for i in range(self.dimension)]

        #to transform this in unit vectors we need to scale the result
        scale=sum(n*n for n in random_p)**-0.5

        return tuple(c*scale for c in random_p)

    def noise(self, *point):
        if self.dimension != len(point):
            raise ValueError(f"The dimension must be equal to {len(point)}, got  {self.dimension}")

        #getting the coordinates of the closest corners in the grid to point given
        grid_corners= []
        for c in point:
            min_coord= math.floor(c)
            max_coord= min_coord+1
            grid_corners.append((min_coord,max_coord))

        #calculate the dot product of each gradient vector with the point given
        dots=[]
        for grid_point in product(*grid_corners):
            if grid_point not in self.gradient:
                self.gradient[grid_point]=self.generate_gradient()
            gradient=self.gradient[grid_point]

            dot=0

            for i in range(self.dimension):
                dot+=gradient[i]*(point[i]-grid_point[i])

            dots.append(dot)


        dim= self.dimension
        while len(dots) > 1:
            dim-=1
            s = smoothstep(point[dim]-grid_corners[dim][0])
            new_dots=[]
            while dots:
                new_dots.append(lerp(dots.pop(0), dots.pop(0), s))

            dots=new_dots

        return dots[0]*self.scale_factor

    # this method can be call like this instance_of_the_class_PerlinNoise(point)
    def __call__(self, *point):
        ret=0
        for oct in range(self.octaves):
            oct2= 1 << oct
            n_points=[]
            for i, coord in enumerate(point):
                coord*= oct2
                if self.tile[i]:
                    coord%=self.tile[i]*oct2
                n_points.append(coord)
            ret+=self.noise(*n_points) /oct2

        #since we add many noises (depends on the number of octaves) we need to scale it back to [-1,1]
        ret/= 2-2**(1-self.octaves)

        if self.unbias:
            #make the ret fall in the [0,1] range
            r=(ret+1)/2
            # the number of times we do this is chosen by hand
            for _ in range(int(self.octaves/2+0.5)):
                r=smoothstep(r)
            # make r fall in the range [-1,1]
            ret = r*2 -1

        return ret
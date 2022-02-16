from shapely.geometry import Point, Polygon

import random

import pulp

import networkx as nx

import math

import h3

class Hexagon(object):
    
    '''
    Functions:
    
    (1) convert geographical coordinates to hexgon
    
    (2) Generate a set of neighbor hexagons for the given hexgon
    
    '''
    
    def __init__(self,resolution):
        
        self.resolution=resolution
        
    def Get_Hex(self,lat,lng):
        
        '''
        Input: latitude,longitude
        
        Output: hexgon
        
        '''
        
        return h3.geo_to_h3(lat,lng,self.resolution)
    
    def Get_Neigh(self,hexgon,radius):
        
        '''
        Input: central hexgon,radius
        
        Output: set of neighbor hexgons
        
        '''
        
        y= h3.hex_range_distances(hexgon, radius)
        
        x=list()
        for y_ in y:
            for y__ in y_:
                x.append(y__)
        return x
    
    def Compact_lists(self,hexagons):
        
        '''
        Input: list of hexgons
        
        Output: set of hexgons
        
        '''
        
        result=list()
    
        for a in hexagons:

            result=list(set(result+a))

        return result
    
    def Center_point(self,hexagon):
        
        return [p for p in h3.h3_to_geo(hexagon)]
    
    def Sample_point(self,hexagon):
        
        '''
        Input: hexgon
        
        Output: point
        
        '''
        
        boundary=h3.h3_to_geo_boundary(hexagon)
        poly = Polygon(boundary)
        (minx, miny, maxx, maxy) = poly.bounds
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p):
                return [list(p.coords)[0][0],list(p.coords)[0][1]]
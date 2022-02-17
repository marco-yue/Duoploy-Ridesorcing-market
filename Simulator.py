#!/usr/bin/env python
# coding: utf-8

# ## 01 Basic packages

# In[1]:


'''Dataframe and matrix'''

import pandas as pd

import numpy as np


'''Spatial analysis'''

import geopandas as gp

from shapely.geometry import Point, Polygon

import networkx as nx

import folium

import h3


'''Mathmatical calculation'''

import math

import random

import pulp

'''Systemic tools'''

import copy

import os, sys


# ## 02 Spatial methods

# In[2]:


'''Get travel distance'''

def Get_distance(point1,point2):
    
    return Point(point1).distance(Point(point2))*111000*1.2

'''Get travel time'''

def Get_travel_time(dis,speed):
    
    return int(dis/speed)

'''Hexagonal classes'''

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


# ## 03 Optimization matching

# In[3]:


'''Order dispatching algorithm'''

def One2One_Matching(Utility):

    '''Define the problem'''

    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    '''Construct our decision variable lists'''

    X = pulp.LpVariable.dicts("X",((O, D) for O in Utility.keys() for D in Utility[O].keys()),lowBound=0,upBound=1,cat='Integer')

    '''Objective Function'''

    model += (pulp.lpSum([Utility[O][D] * X[(O, D)] for O in Utility.keys() for D in Utility[O].keys()]))
    
    '''Each order can only be assigned one driver'''

    for O in Utility.keys():

        model += pulp.lpSum([X[(O, D)] for D in Utility[O].keys()]) <=1

    '''Each driver can only serve one order'''
    
    Utility_={}
    
    for O in Utility.keys():
        for D in Utility[O].keys():
            if D not in Utility_.keys():
                Utility_[D]={O:Utility[O][D]}
            else:
                Utility_[D][O]=Utility[O][D]

    for D in Utility_.keys():

         model += pulp.lpSum([X[(O, D)] for O in Utility_[D].keys()]) <=1

    model.solve()

    result={}

    for var in X:

        var_value = X[var].varValue
        
        if var_value !=0:
            
            result[var[0]]=var[1]
    

    return result   

'''Matching algorithm'''

def One2Many_Matching(Utility,Order_quantity,Dropoff_Drivers):

    '''Define the problem'''

    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    '''Construct our decision variable lists'''

    X = pulp.LpVariable.dicts("X",((D,G) for D in Utility.keys() for G in Utility[D].keys()),lowBound=0,upBound=1,cat='Integer')

    '''Objective Function'''

    model += (pulp.lpSum([Utility[D][G] * X[(D, G)] for D in Utility.keys() for G in Utility[D].keys()]))
    
    '''Each Driver '''

    for D in Utility.keys():

        model += pulp.lpSum([X[(D, G)] for G in Utility[D].keys()]) <= 1

    '''Each Hexagon'''
    
    Utility_={}
    
    for D in Utility.keys():
        for G in Utility[D].keys():
            if G not in Utility_.keys():
                Utility_[G]={D:Utility[D][G]}
            else:
                Utility_[G][D]=Utility[D][G]

    for G in Utility_.keys():

        model += pulp.lpSum([X[(D, G)] for D in Utility_[G].keys()]) <= 10*Order_quantity[G]

    model.solve()

    result={}

    for var in X:

        var_value = X[var].varValue
        
        if var_value !=0:
            
            result[var[0]]=var[1]

    return result



# ## 05 Basic parameters

# In[46]:


'''Systemic paths'''

Load_path='./NYC_Network/'



'''Parameters'''

Start_step=2520 # Start stamp 

End_step=3600 # End stamp 

# Max_waiting=12 # Waiting Patience

radius=2000 # Matching Radius

grid_radius=int(np.ceil(radius/350.0)) # Matching Radius in hexagonal level

speed=20000/360 # Speed, meters per ten seconds 

resolution = 9

ini_step=2520



'''Network data'''

'''Grid-related data'''

Grid_list=np.load(os.path.join(Load_path,'Grids.npy'),allow_pickle=True)

Grid_Point=np.load(os.path.join(Load_path,'Grid_Point.npy'),allow_pickle=True).item()

'''Point-related data'''

Points_list=np.load(os.path.join(Load_path,'Points_list.npy'),allow_pickle=True)

Link_Point=np.load(os.path.join(Load_path,'Link_Point.npy'),allow_pickle=True).item()

Point_coordinate=np.load(os.path.join(Load_path,'Point_coordinate.npy'),allow_pickle=True).item()

Point_Grid=np.load(os.path.join(Load_path,'Point_Grid.npy'),allow_pickle=True).item()


'''Initialize classes'''

Hex=Hexagon(resolution)


'''date'''

day='2020-12-01'

'''Wage'''

Commission_rate={'A':0.20,'B':0.25}

'''Drivers utility ''' 


beta1=0.15

decline_utility=5


'''Drivers utility'''

beta2=0.2


# In[ ]:


'''(1) Load Order data'''

Order_df=pd.read_csv(os.path.join('Order_df_'+str(day)+'.csv'))

Order_df=Order_df.drop(columns=['Unnamed: 0'])


Order_df['Travel_duration']=Order_df.apply(lambda x:int(x['Travel_time']/6),axis=1)

Order_df['Pickup_Point']=Order_df.apply(lambda x:[x['Pickup_Latitude'],x['Pickup_Longitude']],axis=1)

Order_df['Dropoff_Point']=Order_df.apply(lambda x:[x['Dropoff_Latitude'],x['Dropoff_Longitude']],axis=1)

Order_df['Platform']='Null'

'''(2) Load Driver data'''

Driver_df=pd.read_csv(os.path.join('Driver_df_'+str(day)+'.csv'))

Driver_df=Driver_df.drop(columns=['Unnamed: 0'])

Driver_df['Point']=Driver_df.apply(lambda x:[float(p) for p in x['Point'].replace('[','').replace(']','').split(',')],axis=1)

Driver_df['Reposition_Point']=Driver_df.apply(lambda x:x['Point'],axis=1)


'''(3) Simulator'''

Record= pd.DataFrame([],columns=['step','Driver_selelction_A', 'Driver_selelction_B',                                 'Passenger_selection_A','Passenger_selection_B',                                 'Instant_match','Instant_cancel','Idle_vehicles',                                 'Cumulative_match', 'Cumulative_cancel'],dtype=object)


for step in range(Start_step,End_step,1):
    
    print('*'*50)
    
    print('Current step ',step)
    
    '''(3-1) collect unserved orders and arriving orders'''

    Order_batch=Order_df[(Order_df['Arrive_step']<=step)&(Order_df['Driver_id']=='Waiting')]

    Order_info={}

    for idx,row in Order_batch.iterrows():
        
        Order_info[row['Order_id']]={}
        
        Order_info[row['Order_id']]['Pickup_Point']=row['Pickup_Point']
        
        Order_info[row['Order_id']]['Pickup_Grid']=row['Pickup_Grid']
        
        Order_info[row['Order_id']]['Dropoff_Point']=row['Dropoff_Point']
        
        Order_info[row['Order_id']]['Dropoff_Grid']=row['Dropoff_Grid']
        
        Order_info[row['Order_id']]['Travel_time']=row['Travel_time']
        
        Order_info[row['Order_id']]['Travel_duration']=row['Travel_duration']

        Order_info[row['Order_id']]['Match_Grids']=Hex.Get_Neigh(row['Pickup_Grid'],grid_radius)

        Order_info[row['Order_id']]['Matching_patience']=row['Matching_patience']

        Order_info[row['Order_id']]['Fare_A']=row['Fare_A']
        
        Order_info[row['Order_id']]['Fare_B']=row['Fare_B']

    Operation_Grids=Hex.Compact_lists([x['Match_Grids'] for x in Order_info.values()])
    
    
    '''(3-2) collect Idle drivers'''
    
    Driver_batch=Driver_df[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle')&(Driver_df['Grid'].isin(Operation_Grids))]

    Driver_info={}

    Grid_Drivers={}

    for idx,row in Driver_batch.iterrows():
        
        Driver_info[row['Driver_id']]={}

        Driver_info[row['Driver_id']]['Point']=row['Point']
        
        Driver_info[row['Driver_id']]['Grid']=row['Grid']
        
        if row['Grid'] in Grid_Drivers.keys():

            Grid_Drivers[row['Grid']].append(row['Driver_id'])

        else:

            Grid_Drivers[row['Grid']]=[row['Driver_id']]
            
    Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
    
            
    '''(3-3) Order Dispatching'''
    
    Matching_result={}
    
    Pickup_time={order_id:{} for order_id in Order_info.keys()}

    
    Profit_A={order_id:{} for order_id in Order_info.keys()}
    
    Profit_B={order_id:{} for order_id in Order_info.keys()}

    for order_id in Order_info.keys():
        
        origin=Order_info[order_id]['Pickup_Point']

        Fare_A=Order_info[order_id]['Fare_A']
        
        Fare_B=Order_info[order_id]['Fare_B']
        
        Candidate_grids=Order_info[order_id]['Match_Grids']

        for grid in Candidate_grids:

            '''Existing driver?'''
            
            if grid in Grid_Drivers.keys():

                for driver_id in Grid_Drivers[grid]:

                        Profit_A[order_id][driver_id]=Commission_rate['A']*Fare_A
                        
                        Profit_B[order_id][driver_id]=Commission_rate['B']*Fare_B
                        
                        loc=Driver_info[driver_id]['Point']

                        pickup_dis=Get_distance(origin,loc)
                        
                        Pickup_time[order_id][driver_id]=int(Get_travel_time(pickup_dis,speed))
                        
    Matching_result['A']=One2One_Matching(Profit_A)
    
    Matching_result['B']=One2One_Matching(Profit_B)
    
    
    
    
    
    '''(3-4) Drivers make a selection'''
    
    Drivers_Utility={}
    
    for platform,results in Matching_result.items():
        
        for order_id,driver_id in results.items():
            
            if driver_id not in Drivers_Utility.keys():

                Drivers_Utility[driver_id]={}
                
                Drivers_Utility[driver_id][platform+'_'+order_id]=Order_info[order_id]['Fare_'+platform]*(1-Commission_rate[platform])-beta1*Order_info[order_id]['Travel_duration']
                
            else:
                
                Drivers_Utility[driver_id][platform+'_'+order_id]=Order_info[order_id]['Fare_'+platform]*(1-Commission_rate[platform])-beta1*Order_info[order_id]['Travel_duration']
                
    for driver_id in Drivers_Utility.keys():
                
        Drivers_Utility[driver_id]['Decline']=decline_utility
        
        
    Drivers_Selection={}
    
    Drivers_Selection_Count={'A':0,'B':0}
    
    for driver_id in Drivers_Utility.keys():
        
        selection=max(Drivers_Utility[driver_id], key=Drivers_Utility[driver_id].get)
        
        platform=selection.split('_')[0]
        
        Drivers_Selection[driver_id]=selection
        
        if platform!='Decline':
        
            Drivers_Selection_Count[platform]+=1
    
    '''(3-5) Passengers make a selection'''
    
    Passengers_Utility={}
    
    for platform,results in Matching_result.items():
        
        for order_id,driver_id in results.items():
            
            if order_id not in Passengers_Utility.keys():

                Passengers_Utility[order_id]={}
                
                Passengers_Utility[order_id][driver_id+'_'+platform]=Order_info[order_id]['Fare_'+platform]-beta2*Order_info[order_id]['Travel_duration']
                
            else:
                
                Passengers_Utility[order_id][driver_id+'_'+platform]=Order_info[order_id]['Fare_'+platform]-beta2*Order_info[order_id]['Travel_duration']
            
        
        
    Passengers_Selection={}
    
    Passengers_Selection_Count={'A':0,'B':0}
    
    for order_id in Passengers_Utility.keys():
        
        selection=min(Passengers_Utility[order_id], key=Passengers_Utility[order_id].get)
        
        platform=selection.split('_')[-1]
        
        Passengers_Selection[order_id]=selection
        
        Passengers_Selection_Count[platform]+=1
    
        
    '''(3-6) Consensus'''
    
    Matching_result={}
    
    Matching_platform={}
    
    Driver_consensus=[]
    
    Order_consensus=[]
    
    for driver_id,value in Drivers_Selection.items():
        
        Driver_consensus.append(driver_id+'_'+value)
        
    for order_id,value in Passengers_Selection.items():
        
        Order_consensus.append(value+'_'+order_id)
        
    Consensus=[pair for pair in Driver_consensus if pair in Order_consensus]
    
    for pair in Consensus:
        
        order_id=pair.split('_')[2]
        
        driver_id=pair.split('_')[0]
        
        platform=pair.split('_')[1]
        
        Matching_result[order_id]=driver_id
        
        Matching_platform[order_id]=platform
        


    
    
    '''(3-7) Assignment'''

    
    for order_id,driver_id in Matching_result.items():
        
        Added_item={}
        
        Pickup_step=step+Pickup_time[order_id][driver_id]
        
        Dropoff_step=Pickup_step+Order_info[order_id]['Travel_time']

        Order_df.loc[(Order_df['Order_id']==order_id),'Response_step']=step

        Order_df.loc[(Order_df['Order_id']==order_id),'Pickup_step']=Pickup_step

        Order_df.loc[(Order_df['Order_id']==order_id),'Driver_id']=driver_id

        Order_df.loc[(Order_df['Order_id']==order_id),'Platform']=Matching_platform[order_id]

        '''Matched driver'''

        Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id']==driver_id),'Order_id']=order_id

        Added_item['Driver_id']=driver_id

        Added_item['Order_id']='Idle'

        Added_item['Step']=Dropoff_step

        Added_item['Point']=Order_info[order_id]['Dropoff_Point']

        Added_item['Grid']=Order_info[order_id]['Dropoff_Grid']

        Added_item['Reposition_Point']=Order_info[order_id]['Dropoff_Point']

        Added_item['Reposition_Grid']=Order_info[order_id]['Dropoff_Grid']
        
        Driver_df=Driver_df.append(Added_item, ignore_index=True)


            
    '''Canceled Orders'''
        
    Unmatched_orders=[O for O in Order_info.keys() if O not in Matching_result.keys()]
    
    cancellation=0

    if len(Unmatched_orders)!=0:
        
        cancellation=Order_df.loc[((step-Order_df['Arrive_step'])>Order_df['Matching_patience'])&(Order_df['Order_id'].isin(Unmatched_orders))].shape[0]

        Order_df.loc[((step-Order_df['Arrive_step'])>Order_df['Matching_patience'])&(Order_df['Order_id'].isin(Unmatched_orders)),'Driver_id']='Canceled'

        
    '''(3-8) Repositioning'''
    
    Repositioned_drivers=copy.deepcopy(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle')])
    
    Repositioned_drivers['Next_Step']=step+1

    Repositioned_drivers=Repositioned_drivers[['Driver_id','Order_id','Next_Step','Point','Grid','Reposition_Point','Reposition_Grid']]

    Repositioned_drivers=Repositioned_drivers.rename(columns={'Next_Step':'Step'})

    Repositioned_drivers=Repositioned_drivers[['Driver_id','Order_id','Step','Point','Grid','Reposition_Point','Reposition_Grid']]
    
    Driver_df=pd.concat([Driver_df,Repositioned_drivers],ignore_index=True)
    
    
    '''(3-8) Statistics'''
    
    Instant_cancel=cancellation
    
    Instant_match=len(Matching_result)
    
    Culmulative_cancel=Order_df.loc[(Order_df['Driver_id']=='Canceled')].shape[0]
        
    Culmulative_match=Order_df.loc[(Order_df['Driver_id']!='Waiting')&(Order_df['Driver_id']!='Canceled')].shape[0]
    
    
    record={}
    
    record['step']=step
    
    record['Driver_selelction_A']=Drivers_Selection_Count['A']
    
    record['Driver_selelction_B']=Drivers_Selection_Count['B']
    
    record['Passenger_selection_A']=Passengers_Selection_Count['A']
    
    record['Passenger_selection_B']=Passengers_Selection_Count['B']
    
    record['Instant_match']=Instant_match
    
    record['Instant_cancel']=Instant_cancel
    
    record['Idle_vehicles']=Repositioned_drivers.loc[Repositioned_drivers['Point']==Repositioned_drivers['Reposition_Point']].shape[0]

    record['Cumulative_match']=Culmulative_match
    
    record['Cumulative_cancel']=Culmulative_cancel
    
    Record=Record.append(record, ignore_index=True)
    

    print('\n'*2)
    
    
    print('Instant match: ',Instant_match)
    
    print('Instant cancel: ',Instant_cancel)
    
    print('Cumulative match: ',Culmulative_match)
    
    print('Cumulative cancel: ',Culmulative_cancel)
    
    
        


# In[ ]:


Order_df.to_csv(os.path.join('Order_result_'+str(day)+'.csv'))

Driver_df.to_csv(os.path.join('Driver_result_'+str(day)+'.csv'))

Record.to_csv(os.path.join('Record_'+str(day)+'.csv'))


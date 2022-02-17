#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd

from shapely.geometry import Point, Polygon

import numpy as np

import random

import pulp

import networkx as nx

import math

import h3

import os,sys

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import copy

sys.path.append('./Class/')

from Hexagon import Hexagon

from sklearn.utils import shuffle

import scipy.stats as stats


# ## 01 Parameters

# In[42]:


'''Systemic paths'''

Load_path='./NYC_Network/'

'''day'''

day='2020-12-01'


'''Time step'''

Start_step=2520 # Start stamp 

End_step=3600 # End stamp 

'''Matching radius'''

radius=2000 # Matching Radius

grid_radius=int(np.ceil(radius/350.0)) # Matching Radius in hexagonal level

'''Speed'''

speed=20000/360 # Speed, meters per ten seconds 

resolution = 9

'''Network data'''

'''Grid-related data'''

Grid_list=np.load(os.path.join(Load_path,'Grids.npy'),allow_pickle=True)

Grid_Point=np.load(os.path.join(Load_path,'Grid_Point.npy'),allow_pickle=True).item()

'''Point-related data'''

Points_list=np.load(os.path.join(Load_path,'Points_list.npy'),allow_pickle=True)

Link_Point=np.load(os.path.join(Load_path,'Link_Point.npy'),allow_pickle=True).item()

Point_coordinate=np.load(os.path.join(Load_path,'Point_coordinate.npy'),allow_pickle=True).item()

Point_Grid=np.load(os.path.join(Load_path,'Point_Grid.npy'),allow_pickle=True).item()

'''Hexagonal region'''

resolution=9

Hex=Hexagon(resolution)

'''Systematic parameters'''

Commission_rate={'A':0.20,'B':0.25}

beta1=1.2

beta2=0.6

beta3=0.4

beta4=0.6




# ## 02 Order disatching algorithm

# In[43]:


'''Order dispatching algorithm'''

def One2One_Matching(Utility):
    
    '''Shuffle Array'''
    
    Orders=list(Utility.keys())
    
    Drivers={}
    
    for O in Orders:
        
        array=list(Utility[O].keys())
        
        Drivers[O]=shuffle(array)

    '''Define the problem'''

    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    '''Construct our decision variable lists'''

    X = pulp.LpVariable.dicts("X",((O, D) for O in Orders for D in Drivers[O]),lowBound=0,upBound=1,cat='Integer')

    '''Objective Function'''

    model += (pulp.lpSum([Utility[O][D] * X[(O, D)] for O in Orders for D in Drivers[O]]))
    
    '''Each order can only be assigned one driver'''

    for O in Orders:

        model += pulp.lpSum([X[(O, D)] for D in Drivers[O]]) <=1

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
                              
    '''Solve the problem'''

    model.solve()

    result={}

    for var in X:

        var_value = X[var].varValue
        
        if var_value !=0:
            
            result[var[0]]=var[1]
    

    return result
                              
        


# ## 03 Simulator

# In[45]:


'''Get travel distance'''

def Get_distance(point1,point2):
    
    return Point(point1).distance(Point(point2))*111000*1.2

'''Get travel time'''

def Get_travel_time(dis,speed):
    
    return int(dis/speed)

'''Truncated Gaussian distribution'''

def Truncated_Gauss(mu,sigma,lower,upper):

    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    
    return [x for x in X.rvs(1)][0]

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
    
    print('Number of waiting orders: ',len(Order_info))
    
    
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
    
    print('Number of idle drivers: ',len(Idle_drivers))
    
   
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
        
        commission_rate=Commission_rate[platform]
        
        for order_id,driver_id in results.items():
            
            fare=Order_info[order_id]['Fare_'+platform]
            
            pickup_time=Pickup_time[order_id][driver_id]
            
            travel_time=Order_info[order_id]['Travel_duration']
            
            utility=(1-commission_rate)*fare-Truncated_Gauss(0.4,0.1,0.3,0.5)*int(pickup_time/6)
            
            if driver_id not in Drivers_Utility.keys():

                Drivers_Utility[driver_id]={}
                
                Drivers_Utility[driver_id][platform+'_'+order_id]=utility
                
                Drivers_Utility[driver_id]['Decline']=Truncated_Gauss(0.3,0.1,0.2,0.4)*travel_time
                
            else:
                
                Drivers_Utility[driver_id][platform+'_'+order_id]=utility
                

    '''Probability'''
    
    Driver_prob={}
    
    for driver_id in Drivers_Utility.keys():
        
        exp_sum=sum([np.exp(u) for u in Drivers_Utility[driver_id].values()])
        
        Driver_prob[driver_id]={}
        
        for order_id in Drivers_Utility[driver_id].keys():
            
            exp_v=np.exp(Drivers_Utility[driver_id][order_id])
            
            Driver_prob[driver_id][order_id]=exp_v/exp_sum
            
    '''Selection'''
        
    Drivers_Selection={}
    
    Drivers_Selection_Count={'A':0,'B':0,'Decline':0}
    
    for driver_id in Driver_prob.keys():
        
        selection=np.random.choice(list(Driver_prob[driver_id].keys()), p=list(Driver_prob[driver_id].values()))
        
        platform=selection.split('_')[0]
        
        Drivers_Selection[driver_id]=selection
        
        Drivers_Selection_Count[platform]+=1
            
    
    '''(3-5) Passengers make a selection'''
    
    Passengers_Utility={}
    
    for platform,results in Matching_result.items():
        
        for order_id,driver_id in results.items():
            
            fare=Order_info[order_id]['Fare_'+platform]
            
            pickup_time=Pickup_time[order_id][driver_id]
            
            travel_time=Order_info[order_id]['Travel_duration']
            
            utility=-1*fare-int(pickup_time/6)*Truncated_Gauss(0.6,0.2,0.4,0.8)
            
            
            if order_id not in Passengers_Utility.keys():

                Passengers_Utility[order_id]={}
                
                Passengers_Utility[order_id][driver_id+'_'+platform]=utility
                
                Passengers_Utility[order_id]['Decline']=-1*Truncated_Gauss(1.5,0.5,1.0,2.0)*travel_time
                
            else:
                
                Passengers_Utility[order_id][driver_id+'_'+platform]=utility
                
  
        
    '''Probability'''
    
    Passenger_prob={}
    
    for order_id in Passengers_Utility.keys():
        
        exp_sum=sum([np.exp(v) for v in Passengers_Utility[order_id].values()])
        
        Passenger_prob[order_id]={}
        
        for driver_id in Passengers_Utility[order_id].keys():
            
            exp_v=np.exp(Passengers_Utility[order_id][driver_id])
            
            Passenger_prob[order_id][driver_id]=exp_v/exp_sum
        
    '''Selection'''
        
    Passengers_Selection={}
    
    Passengers_Selection_Count={'A':0,'B':0,'Decline':0}
    
    for order_id in Passengers_Utility.keys():
        
        selection=np.random.choice(list(Passenger_prob[order_id].keys()), p=list(Passenger_prob[order_id].values()))
        
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
    
    record['Driver_selelction_Decline']=Drivers_Selection_Count['Decline']
    
    record['Passenger_selection_A']=Passengers_Selection_Count['A']
    
    record['Passenger_selection_B']=Passengers_Selection_Count['B']
    
    record['Passenger_selection_Decline']=Passengers_Selection_Count['Decline']
    
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


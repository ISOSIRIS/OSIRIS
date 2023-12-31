# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 02:26:31 2023

@author: Owner
"""

# -----===========.......    AKA  AGENT  J & K  * REMEMBER, WHAT YOU THINK YOU SAW YOU DID NOT SEE, M I B OR IBM  ???........

import numpy as np
import math
import joblib
from dynamicform import Dynamic_form_wt as Dyf

# Create two NumPy arrays
x = [[-0.15222846,  0.88082343,  0.91212123],
     [-0.91304799,  0.98161611, -0.98122095],
     [-0.32155498, -0.64036311, -0.75064098]]

y =[
          [math.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id))))),math.cos(id(id(id(id(id)))))  ],
          [np.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id))))), np.cos(id(id(id(id(id)))))],
          [math.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id)))))]]

z =[
          [math.cos(id(id(id(id(id))))), np.cos(id(id(id(id(id))))),math.cos(id(id(id(id(id)))))  ],
          [math.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id)))))],
          [np.cos(id(id(id(id(id))))), math.cos(id(id(id(id(id))))), np.cos(id(id(id(id(id)))))]]

# Save the arrays to a NumPy file

np.savez('scn1.npz', x=x, y=y, z=z);np.savez('scn2.npz', x=x, y=y, z=z); np.savez('scn4.npz', x=x, y=y, z=z); np.savez('scn5.npz', x=x, y=y, z=z)
np.savez('scn6.npz', x=x, y=y, z=z); np.savez('scn3.npz', x=x, y=y, z=z); np.savez('scn7.npz', x=x, y=y, z=z); np.savez('scn8.npz', x=x, y=y, z=z)
np.savez('scn9.npz', x=x, y=y, z=z); np.savez('scn10.npz', x=x, y=y, z=z); np.savez('scn11.npz', x=x, y=y, z=z) 

##########----------- Subneuron2
np.savez('scn12.npz', x=x, y=y, z=z);np.savez('scn13.npz', x=x, y=y, z=z); np.savez('scn14.npz', x=x, y=y, z=z); np.savez('scn15.npz', x=x, y=y, z=z)
np.savez('scn16.npz', x=x, y=y, z=z); np.savez('scn17.npz', x=x, y=y, z=z); np.savez('scn18.npz', x=x, y=y, z=z); np.savez('scn19.npz', x=x, y=y, z=z)
np.savez('scn20.npz', x=x, y=y, z=z); np.savez('scn21.npz', x=x, y=y, z=z); np.savez('scn22.npz', x=x, y=y, z=z) 

##########----------- Subneuron3
np.savez('scn23.npz', x=x, y=y, z=z); np.savez('scn24.npz', x=x, y=y, z=z); np.savez('scn25.npz', x=x, y=y, z=z); np.savez('scn26.npz', x=x, y=y, z=z)

##########----------- Subneuron4
np.savez('scn27.npz', x=x, y=y, z=z); np.savez('scn28.npz', x=x, y=y, z=z); np.savez('scn29.npz', x=x, y=y, z=z); np.savez('scn30.npz', x=x, y=y, z=z)

##########----------- Subneuron5
np.savez('scn31.npz', x=x, y=y, z=z); np.savez('scn32.npz', x=x, y=y, z=z); np.savez('scn33.npz', x=x, y=y, z=z); np.savez('scn34.npz', x=x, y=y, z=z)




# Initialize a 3D array of shape (9, 9, 9) with random values
weight_array = np.random.randn(9, 9, 9)                            #list of lists so im told...
'''
# Initialize a 2D input array of shape (3, 3) with random values
input_array = np.random.randn(3, 3)

# Expand dimensions of the input array to (3, 3, 1)
expanded_input = input_array[:, :, np.newaxis]

# Reshape the expanded input array to (9, 9, 9)
expanded_input = expanded_input.reshape((9))

# Perform element-wise multiplication with broadcasting
result = weight_array * expanded_input
'''
np.savez('Subc.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc3.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc1.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc4.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc2.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc5.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc6.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc9.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc7.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('SubcTEN.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc8.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc11.npz', x=weight_array , y= weight_array, z= weight_array)

##########---------- Subneuron2
np.savez('Subc12.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc13.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc14.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc15.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc16.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc17.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc18.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc19.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc20.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc21.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc22.npz', x=weight_array , y= weight_array, z= weight_array);np.savez

##########---------- Subneuron3
np.savez('Subc23.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc24.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc25.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc26.npz', x=weight_array , y= weight_array, z= weight_array)

##########---------- Subneuron4
np.savez('Subc27.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc28.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc29.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc30.npz', x=weight_array , y= weight_array, z= weight_array)

##########---------- Subneuron5
np.savez('Subc31.npz',  x=weight_array , y= weight_array, z= weight_array);np.savez('Subc32.npz', x=weight_array , y= weight_array, z= weight_array)
np.savez('Subc33.npz', x=weight_array , y= weight_array, z= weight_array);np.savez('Subc34.npz', x=weight_array , y= weight_array, z= weight_array)






#  MINI NEURALIZER FOR THE J'SSS
def mini_j_neuralizer():      
         ######## INIT__ BLOCK TOP___########
        ###########################################
        
        y = np.array([[-1.15222846,  2.88082343,  9.91212123],
             [-0.91304799,  0.98161611, -0.98122095],
             [ 8.32155498, -0.64036311, -1.75064098]]) # *omega
        home = y
        
        core1 = y
        core2 = y
        core3 = y
        
        cybi1 = Dyf()
        cybi2 = Dyf()
        cybi3 = Dyf()
        
        weight_array = np.random.randn(9, 9, 9)   #*omega     
        
        cube1 = weight_array
        cube2 = weight_array
        cube3 = weight_array
        
        iam = {'home':home, 'wt1':core1,'wt2':core2,'wt3':core3,'cube1':cube1,'cube2':cube2,'cube3':cube3,'cybi1':cybi1, 'cybi2':cybi2, 'cybi3':cybi3}
        joblib.dump(iam, 'Jwts')
        joblib.dump(iam, 'Jwts1')
        joblib.dump(iam, 'Jwts2')
        joblib.dump(iam, 'Jwts3')
        joblib.dump(iam, 'Jwts4')
        joblib.dump(iam, 'Jwts5')
   
        
        
        
mini_j_neuralizer()









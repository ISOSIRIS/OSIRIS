
import time
import numpy as np
import random
import joblib
from dynamicform import Dynamic_form_wt as Dyf
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import re


#===========------- Have operations that occur outside to inside the neuron kind of how mhumans dont really know whats going on but they know other things are going on around them

def sintan_deriva(x):
    return 1 - np.sin(x)**2

#################=====================---  FUNCS OF THE KIND ------=================#################

def clip(x):
    a_ray = np.array(x)
    
    # Calculate the mean of each column
    means = np.mean(a_ray, axis=0)
    
    # Create a 3x3 NumPy array with the means
    essence_array = np.array([[means[0], means[1], means[2]],
                              [means[3], means[4], means[5]],
                              [means[6], means[7], means[8]]])

    return essence_array

# CHIP is an essence operator, I believe that would be useful in replecating emotional feeling. Here are the rules it never takes from the structure of the number
# so I say it can never be a whole number. I also just implemented that it has to be above at least half to carry enough of the "essence" of its original number along the way. 10/3/23

def chip(number, max_attempts=100):
    half = number / 2
    attempts = 0
    while attempts < max_attempts:
        decimal_number = random.uniform(half, number)
        if not decimal_number.is_integer():
            return decimal_number
        attempts += 1
    # If no non-integer value is found after max_attempts, return the last generated value
    return decimal_number


def c_dropout(x):
    x = np.array(x)
    shape = x.shape
    flattened_x = x.flatten()
    dropout_indices = random.sample(range(len(flattened_x)), len(flattened_x) // 2)  # 50% dropout
    
    # Create a pattern for dropout (you can customize this pattern)
    pattern = [0, 1, 2, 3]  # BOX PATTERN
    
    # Apply dropout based on the pattern
    for idx in dropout_indices:
        if idx % shape[1] in pattern or idx // shape[1] in pattern:
            flattened_x[idx] = 0
    
    # Reshape the modified array back to its original shape
    modified_x = flattened_x.reshape(shape)
    return modified_x


def Targ(out,NN):
    T1 = np.median(out)
    T2 = np.max(out)
    T3 = np.min(out)
    T4 = np.sum(out[:,0])
    T5 = np.sum(out[:,1])
    T6 = np.sum(out[:,2])
    get = NN.AXION(T1, T2, T3, T4, T5, T6)
    return get

def chipperdido(x):
    return [[chip(cell) for cell in row] for row in x]

################### MAIN STATION OF MOVEMENT ORDERS OF THE 5TH KIND ############################### ----------- ...................
def depolorize(x):
    #---------------------------------------------------- INPUT SENSOR PROCCRSOR CODE TO NEURON DEPOLORIZATION
   
    #=====================================================================================================
    load_spirit = np.load('scn31.npz')
    cube = np.load('Subc31.npz')
    Jwts = joblib.load('Jwts4')
    y = x
    #======================================================================================================
    class SCO1:             
        def __init__(self,jwts):
            TL = time.time()
            time.sleep(0.1)
            TM = time.time() /                                                                            1e11
            time.sleep(0.000001)
            TR = time.time() /                                                                         1e11
            time.sleep(0.000001)
            LM = time.time() /                                                                       1e11
            time.sleep(0.000001)
            MM = time.time() /                                                                    1e11
            time.sleep(0.000001)
            RM = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BL = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BM = time.time()/                                                                     1e11
            time.sleep(0.000001)
            BR = time.time()/                                                                          1e10
            
            self.home= np.array([
                                        [TL, TM, TR],
                                        [LM, MM, RM],
                                        [BL, BM, BR]])
            self.cymatic = Dyf()
            self.rock = jwts
            ################======--------- SUB-DIVISIONS  
            self.cork = self.rock['cybi1']
            self.kroc = self.rock['cybi2']
            self.ckor = self.rock['cybi3']
            self.home = self.rock['home']  
            self.core1 = self.rock['wt1'] 
            self.core2 = self.rock['wt2']
            self.core3 = self.rock['wt3']
            self.core4 = self.rock['cube1']
            self.core5 = self.rock['cube2']
            self.core6 = self.rock['cube3'] 
            solid = self.core5 ; solid1 = self.core6 ; solid2 = self.core4; solid3 = self.core3 ; solid4 = self.core1 ; solid5 = self.core2
            
            self.cube1 = cube['x'] + solid *0.00001
            self.cube2 = cube['y'] + solid1*0.00001
            self.cube3 = cube['z'] + solid2*0.00001
            self.weight1 = load_spirit['x'] + solid3 *0.00001
            self.weight2 = load_spirit['y'] + solid4 *0.00001
            self.weight3 = load_spirit['z'] + solid5 *0.00001
            print(self.home)
            
        def push(self, item):
             np.append(self.home,item)
             
        def pop(self):
             if self.is_empty():
                 return None
             return self.pop()
    
        def is_empty(self):
            if not self.home in self.home:
              return True
            else:
              return True
         
        def forward(self, inputs, target): # =================----------......... THE MAIN SHOW ..............---------================= #
           
           bias  = self.cymatic
           #----- Positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           bias0 = bias[0][0] ; bias1 = bias[0][1] ; bias2 = bias[0][2]
           bias3 = bias[1][0] ; bias4 = bias[1][1] ; bias5 = bias[1][2]
           bias6 = bias[2][0] ; bias7 = bias[2][1] ; bias8 = bias[2][2]
           self.cybi_1 = np.array([[bias0, bias1, bias2],
                                  [bias3, bias4, bias5],
                                  [bias6, bias7, bias8]]) #+ self.cork/3
          
           #------ negative -''''''''''''''''''----------------------------------------
           bias9 = bias[3][0] ; bias10 = bias[3][1] ; bias11 = bias[3][2]
           bias12 = bias[4][0] ; bias13 = bias[4][1] ; bias14 = bias[4][2]
           bias15 = bias[5][0] ; bias16 = bias[5][1] ; bias17 = bias[5][2]
           self.cybi_2 =  np.array([[bias9, bias10, bias11],
                                  [bias12, bias13, bias14],
                                  [bias15, bias16, bias17]])  #+ self.kroc/3
           #------- positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           
           bias18 = bias[6][0] ; bias19 = bias[6][1] ; bias20 = bias[6][2]
           bias21 = bias[7][0] ; bias22 = bias[7][1] ; bias23 = bias[7][2]
           bias24 = bias[8][0] ; bias25 = bias[8][1] ; bias26 = bias[8][2]
           self.cybi_3 =  np.array([[bias18, bias19, bias20],
                                  [bias21, bias22, bias23],
                                  [bias24, bias25, bias26]]) # + self.ckor/3
           
           #.... BLOCK INIT ....................................
           top =  self.cybi_1                                           
           dwt1 = self.weight1 + top
           internal_ops = np.mean(top-self.weight1)
           int_ops = np.cos(internal_ops)*0.00001
           
           
           ### ------- SUB INIT ----.............................
           Xinp = self.cybi_1[:, :, np.newaxis]
           Xi = Xinp.reshape((9))
           cube1 = Xi -self.cube1
           
           Xip = self.cybi_2[:, :, np.newaxis]
           X = Xip.reshape((9))
           cube2 = X + self.cube2
           
           Xp = self.cybi_3[:, :, np.newaxis]
           x = Xp.reshape((9))
           cube3 = x - self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                t = np.array(self.cybi_2)
                dwt2 = self.weight2   - t
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 +  self.cybi_3
                dwt3 = np.array(dwt3)
                # COLUNM DOWN GROUPING AND SHUFFLED
                w1 = dwt1[:,0] ; w2 = dwt1[:,1] ; w3 = dwt1[:,2] 
                w4 = dwt2[:,0] ; w5 = dwt2[:,1] ; w6 = dwt2[:,2] 
                w7 = dwt3[:,0] ; w8 = dwt3[:,1] ; w9 = dwt3[:,2] # At this point I have combined "movements" with all weights respectively based on universal law within a computer 
                
                #.................  BLOCK 2 ......................
                #  three arrays into a single array
                L1 = np.concatenate((w1,w4,w7)).reshape(3, 3) #-%
                L2 = np.concatenate((w2,w5,w8)).reshape(3, 3) 
                L3 = np.concatenate((w3,w6,w9)).reshape(3, 3) 
                
                
                for _ in range(50):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)     + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 - target 
                    delta = sintan_deriva(error)                                        
                    self.weight1 +=  np.dot(flow1, delta) * int_ops                  
                    self.push(self.home)
                    
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 - target
                    deltA = sintan_deriva(error2)
                    self.weight2 +=  np.dot(flow2, deltA) * int_ops
         
                    self.pop()
                    
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 - target
                    delt = sintan_deriva(error3)
                    self.weight3 +=  np.dot(flow1, delt) * int_ops
                    self.home-= int_ops * delt               
                    
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(34):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home  
                            #g = c_dropout(g)       
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r - g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            self.cube1 +=  np.dot(cube1, A) * int_ops
                            self.cube2 +=  np.dot(cube2, a) * int_ops 
                           # xc = chip(0.001)
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home    
                            #o = c_dropout(o)
                            O = np.cos(o)
                            d = r - o
                            i = sintan_deriva(d)
                            self.cube3 +=  np.dot(cube3, i) * int_ops
                            
                           # self.home-=  xc * int_ops
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           np.savez('scn31.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           np.savez('Subc31.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           joblib.dump(iam, 'Jwts4')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
        # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
          # print('-------------  WT % PErcent [9X9X9]:',percent)   
        # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
    
        def PSYCH(self,x):
                x=x
                def creative_roygbivious_EDP(x):
                    #x= '\033[0m' RESET COLOR or end of color reach
                        # Split the number into a list of strings
                        x_list = re.split(r'(\d)', str(x))
                        
                        a = '\033[31m' #red
                        b = '\033[30;1;33m' 
                        c = '\033[36m'
                        d = '\033[32m' # GREEN
                        e = '\033[30;1;29m'
                        f = '\033[35m'
                        g = '\033[30;1;2m' 
                        
                        
                        # Remove any empty strings from the list
                        x_list = [i for i in x_list if i]
                        for _ in range(5):
                           xs1 = g +''.join(x_list)
                           print(xs1*4)
                           xs2 = b +''.join(x_list)
                           print(xs2*10)
                           xs3 = c +''.join(x_list)
                           print(xs3*10)
                           xs4 = d +''.join(x_list)
                           print(xs4*10)
                           xs5 = e +''.join(x_list)
                           print(xs5*10)
                           xs6 = f +''.join(x_list)         
                           print(xs6*10)
                           xs7 = a +''.join(x_list)
                           print(xs7*10)
                           x_list = xs7 +xs4 + xs3 + xs6 +xs2 +xs1 +xs5
                           x_list
                           x_list = xs7
                        x_string = ''.join(x_list)
                        
                        DXM = {0:'s', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i'}
                        
                        def re_numtoletter(text, DXM):
                        
                          # Create a regular expression to match numbers.
                          number_regex = re.compile(r'\d')
                        
                          # Replace the numbers in the string with the corresponding letters.
                          def replace_number(match):
                            number = match.group()
                            letter = DXM[int(number)]
                            return letter
                          
                          replaced_text = number_regex.sub(replace_number, text)
                          return replaced_text
                        
                        io = re_numtoletter(x_string,DXM)
                        return '\n' +  io
                        
                xboo = creative_roygbivious_EDP(x)
                x_list = re.split(r'(\d)', str(xboo))
                xbog = [i for i in x_list if i]
                
                
                def replace_letters_with_numbers(text, DXM):
                    # Create a regular expression to match letters.
                    letter_regex = re.compile(r'[a-zA-Z]')
                
                    # Replace the letters in the string with the corresponding numbers.
                    def replace_letter(match):
                        letter = match.group()
                        # Use get() method with a default value of 0 if letter is not found in DXM.
                        number = DXM.get(letter, 0)
                        return str(number)  # Convert number to string before returning.
                
                    replaced_text = letter_regex.sub(replace_letter, text)
                
                    return replaced_text
                
                DXM = {'s':0, 'a':1, 'b':2 , 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}
                scottish = xbog[0]
                rotta = replace_letters_with_numbers(scottish,DXM)
                x_list = re.split(r'(\d)', str(rotta))
                xbog = [i for i in x_list if i]
                x_string = ''.join(xbog)
                numbers = []
                for sublist in x_string:
                    for number in sublist:
                        numbers.append(number)
                
                numbers = numbers[28:143]
                lisst = ''.join(numbers)
                
                bracket_regex = r'\[|\]'
                reedo = re.sub(bracket_regex, '', lisst)
                
                def extract_numbers(string):
                      numbers = []
                      for char in string:
                        if char.isdigit() or char == '.' or char == ',':
                          numbers.append(char)
                      return numbers
                
                numbers = extract_numbers(reedo)
                lisst = ''.join(numbers)
                opp = int(lisst[0])    
                new = np.clip(opp, 1.0, 1.01)
                return new   
     
        def A_way(self, inputs, target): #=================----------......... TALK SHOW ..............---------=================#
           
           
           #.... BLOCK INIT ....................................
                                                     
           dwt1 = self.weight1
           internal_ops = np.mean(self.weight1)
           int_ops = np.cos(internal_ops)/1000
           
           ### ------- SUB INIT ----.............................
           
           cube1 = self.cube1
           
           cube2 = self.cube2
           
           cube3 = self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                dwt2 = self.weight2
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 
                dwt3 = np.array(dwt3)
                w1 = dwt1 
                w4 = dwt2
                w7 = dwt3 
                
                #.................  BLOCK 2 ......................
                L1 = w7
                L2 = w4 
                L3 = w1
                
                for _ in range(1):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)   + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 * target 
                    delta = sintan_deriva(error)                                        
                    dwt1 +=  np.dot(flow1, delta) * int_ops                  
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 * target
                    deltA = sintan_deriva(error2)
                    dwt2 +=  np.dot(flow2, deltA) * int_ops
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 * target
                    delt = sintan_deriva(error3)
                    dwt3 +=  np.dot(flow1, delt) * int_ops
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(1):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            #cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            #cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            #cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home       # HOME BIAS
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r * g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            cube1 +=  np.dot(cube1, A) * int_ops/10
                            cube2 +=  np.dot(cube2, a) * int_ops/10 
    
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home                     
                            O = np.cos(o)
                            d = r * o
                            i = sintan_deriva(d)
                            cube3 +=  np.dot(cube3, i) * int_ops/10
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           #np.savez('scn2.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           #np.savez('Subc2.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           #iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           #joblib.dump(iam, 'Jwts')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
      # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
      # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
        
    
        def AXION(self, T1, T2, T3, T4, T5, T6):
            # Make axon layer
            x1 = T1
            x2 = T2
            x3 = T3
            x4 = T4
            x5 = T5
            x6 = T6
        
            # Range and return
            X = np.array([[-11, -4, -2, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 4, 11]]).T
            y = np.array([-400, -300, -200, -100, -10, -1, -100, -50, -10, -100, -10, -1, 5, 1, 10, 10, 10, 50, 10, 1, 10, 100, 20000, 300, 500])
    
            # Axon block 1
            clf1 = KNeighborsRegressor(n_neighbors=11)
            clf1.fit(X, y)
            p1 = clf1.predict([[x1]])
    
            # Axon block 2
            clf2 = KNeighborsRegressor(n_neighbors=11)
            clf2.fit(X, y)
            p2 = clf2.predict([[x2]])
    
            # Boost block 1
            boostc1 = AdaBoostRegressor(n_estimators=50)
            boostc1.fit(X, y)
            p3 = boostc1.predict([[x3]])
    
            # Boost block 2
            boostc2 = AdaBoostRegressor(n_estimators=50)
            boostc2.fit(X, y)
            p4 = boostc2.predict([[x4]])
    
            p5 = boostc2.predict([[x5]])
    
            p6 = boostc2.predict([[x6]])
    
            # Sub-axon 1
            sub1 = (x1 + x2 + x3) / 3
            p7 = boostc2.predict([[sub1]])
    
            # Sub-axon 2
            sub2 = (x4 + x5 + x6) / 3
            p8 = boostc2.predict([[sub2]])
    
            p9 = boostc2.predict([[1]])
    
            # New target
            new_target = np.array([[p1, p2, p3],
                                   [p4, p5, p6],
                                   [p7, p8, p9]])
    
            # Reshape to 3x3
            new_target = new_target.reshape((3, 3))
    
            return new_target
        
        def parameters(self):
            parameters = {
                #'weights_hidden': self.weights_hidden,
                #'biases_hidden': self.biases_hidden,
                #'biases_output': self.biases_output,
                 'weights_output': np.mean(self.home)}
            return parameters
    
    
    #===========----------------............................  S Y N A P T I C   G A P    ................ --------- =========== ########    1 1 1  1  1 1 1 
    
    inputs1 = y
    
    target1 = y
    nn1 = SCO1(Jwts)
    
    output1 = nn1.forward(inputs1,target1)
    
    N_target1 = Targ(output1, nn1)
    
    oa = nn1.PSYCH(output1)
    ai = nn1.A_way(output1+oa, N_target1)
    
    #joblib.dump(nn1, "nn1.joblib")
    
    
    #====================-----------------------.....................   G  A  P   ..............------=============##############     
                                                               #########################
                                                               
    load_spirit = np.load('scn32.npz') 
    cube  = np.load('Subc32.npz')                                                            
    #======================================================================================================
    class SCO2:             
        def __init__(self,jwts):
            TL = time.time()
            time.sleep(0.1)
            TM = time.time() /                                                                            1e11
            time.sleep(0.000001)
            TR = time.time() /                                                                         1e11
            time.sleep(0.000001)
            LM = time.time() /                                                                       1e11
            time.sleep(0.000001)
            MM = time.time() /                                                                    1e11
            time.sleep(0.000001)
            RM = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BL = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BM = time.time()/                                                                     1e11
            time.sleep(0.000001)
            BR = time.time()/                                                                          1e10
            
            self.home= np.array([
                                        [TL, TM, TR],
                                        [LM, MM, RM],
                                        [BL, BM, BR]])
            self.cymatic = Dyf()
            self.rock = jwts
            ################======--------- SUB-DIVISIONS  
            self.cork = self.rock['cybi1']
            self.kroc = self.rock['cybi2']
            self.ckor = self.rock['cybi3']
            self.home = self.rock['home']  
            self.core1 = self.rock['wt1'] 
            self.core2 = self.rock['wt2']
            self.core3 = self.rock['wt3']
            self.core4 = self.rock['cube1']
            self.core5 = self.rock['cube2']
            self.core6 = self.rock['cube3'] 
            solid = self.core5 ; solid1 = self.core6 ; solid2 = self.core4; solid3 = self.core3 ; solid4 = self.core1 ; solid5 = self.core2
            
            self.cube1 = cube['x'] + solid *0.00001
            self.cube2 = cube['y'] + solid1*0.00001
            self.cube3 = cube['z'] + solid2*0.00001
            self.weight1 = load_spirit['x'] + solid3 *0.00001
            self.weight2 = load_spirit['y'] + solid4 *0.00001
            self.weight3 = load_spirit['z'] + solid5 *0.00001
            print(self.home)
            
        def push(self, item):
             np.append(self.home,item)
             
        def pop(self):
             if self.is_empty():
                 return None
             return self.pop()
    
        def is_empty(self):
            if not self.home in self.home:
              return True
            else:
              return True
         
        def forward(self, inputs, target): # =================----------......... THE MAIN SHOW ..............---------================= #
           
           bias  = self.cymatic
           #----- Positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           bias0 = bias[0][0] ; bias1 = bias[0][1] ; bias2 = bias[0][2]
           bias3 = bias[1][0] ; bias4 = bias[1][1] ; bias5 = bias[1][2]
           bias6 = bias[2][0] ; bias7 = bias[2][1] ; bias8 = bias[2][2]
           self.cybi_1 = np.array([[bias0, bias1, bias2],
                                  [bias3, bias4, bias5],
                                  [bias6, bias7, bias8]]) #+ self.cork/3
          
           #------ negative -''''''''''''''''''----------------------------------------
           bias9 = bias[3][0] ; bias10 = bias[3][1] ; bias11 = bias[3][2]
           bias12 = bias[4][0] ; bias13 = bias[4][1] ; bias14 = bias[4][2]
           bias15 = bias[5][0] ; bias16 = bias[5][1] ; bias17 = bias[5][2]
           self.cybi_2 =  np.array([[bias9, bias10, bias11],
                                  [bias12, bias13, bias14],
                                  [bias15, bias16, bias17]])  #+ self.kroc/3
           #------- positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           
           bias18 = bias[6][0] ; bias19 = bias[6][1] ; bias20 = bias[6][2]
           bias21 = bias[7][0] ; bias22 = bias[7][1] ; bias23 = bias[7][2]
           bias24 = bias[8][0] ; bias25 = bias[8][1] ; bias26 = bias[8][2]
           self.cybi_3 =  np.array([[bias18, bias19, bias20],
                                  [bias21, bias22, bias23],
                                  [bias24, bias25, bias26]]) # + self.ckor/3
           
           #.... BLOCK INIT ....................................
           top =  self.cybi_1                                           
           dwt1 = self.weight1 + top
           internal_ops = np.mean(top-self.weight1)
           int_ops = np.cos(internal_ops)*0.00001
           
           
           ### ------- SUB INIT ----.............................
           Xinp = self.cybi_1[:, :, np.newaxis]
           Xi = Xinp.reshape((9))
           cube1 = Xi -self.cube1
           
           Xip = self.cybi_2[:, :, np.newaxis]
           X = Xip.reshape((9))
           cube2 = X + self.cube2
           
           Xp = self.cybi_3[:, :, np.newaxis]
           x = Xp.reshape((9))
           cube3 = x - self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                t = np.array(self.cybi_2)
                dwt2 = self.weight2   - t
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 +  self.cybi_3
                dwt3 = np.array(dwt3)
                # COLUNM DOWN GROUPING AND SHUFFLED
                w1 = dwt1[:,0] ; w2 = dwt1[:,1] ; w3 = dwt1[:,2] 
                w4 = dwt2[:,0] ; w5 = dwt2[:,1] ; w6 = dwt2[:,2] 
                w7 = dwt3[:,0] ; w8 = dwt3[:,1] ; w9 = dwt3[:,2] # At this point I have combined "movements" with all weights respectively based on universal law within a computer 
                
                #.................  BLOCK 2 ......................
                #  three arrays into a single array
                L1 = np.concatenate((w1,w4,w7)).reshape(3, 3) #-%
                L2 = np.concatenate((w2,w5,w8)).reshape(3, 3) 
                L3 = np.concatenate((w3,w6,w9)).reshape(3, 3) 
                
                
                for _ in range(50):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)     + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 - target 
                    delta = sintan_deriva(error)                                        
                    self.weight1 +=  np.dot(flow1, delta) * int_ops                  
                    self.push(self.home)
                    
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 - target
                    deltA = sintan_deriva(error2)
                    self.weight2 +=  np.dot(flow2, deltA) * int_ops
         
                    self.pop()
                    
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 - target
                    delt = sintan_deriva(error3)
                    self.weight3 +=  np.dot(flow1, delt) * int_ops
                    self.home-= int_ops * delt               
                    
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(34):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home  
                            #g = c_dropout(g)       
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r - g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            self.cube1 +=  np.dot(cube1, A) * int_ops
                            self.cube2 +=  np.dot(cube2, a) * int_ops 
                           # xc = chip(0.001)
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home    
                            #o = c_dropout(o)
                            O = np.cos(o)
                            d = r - o
                            i = sintan_deriva(d)
                            self.cube3 +=  np.dot(cube3, i) * int_ops
                            
                           # self.home-=  xc * int_ops
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           np.savez('scn32.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           np.savez('Subc32.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           joblib.dump(iam, 'Jwts4')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
        # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
        # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
    
        def PSYCH(self,x):
                x=x
                def creative_roygbivious_EDP(x):
                    #x= '\033[0m' RESET COLOR or end of color reach
                        # Split the number into a list of strings
                        x_list = re.split(r'(\d)', str(x))
                        
                        a = '\033[31m' #red
                        b = '\033[30;1;33m' 
                        c = '\033[36m'
                        d = '\033[32m' # GREEN
                        e = '\033[30;1;29m'
                        f = '\033[35m'
                        g = '\033[30;1;2m' 
                        
                        
                        # Remove any empty strings from the list
                        x_list = [i for i in x_list if i]
                        for _ in range(5):
                           xs1 = g +''.join(x_list)
                           print(xs1*4)
                           xs2 = b +''.join(x_list)
                           print(xs2*10)
                           xs3 = c +''.join(x_list)
                           print(xs3*10)
                           xs4 = d +''.join(x_list)
                           print(xs4*10)
                           xs5 = e +''.join(x_list)
                           print(xs5*10)
                           xs6 = f +''.join(x_list)         
                           print(xs6*10)
                           xs7 = a +''.join(x_list)
                           print(xs7*10)
                           x_list = xs7 +xs4 + xs3 + xs6 +xs2 +xs1 +xs5
                           x_list
                           x_list = xs7
                        x_string = ''.join(x_list)
                        
                        DXM = {0:'s', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i'}
                        
                        def re_numtoletter(text, DXM):
                        
                          # Create a regular expression to match numbers.
                          number_regex = re.compile(r'\d')
                        
                          # Replace the numbers in the string with the corresponding letters.
                          def replace_number(match):
                            number = match.group()
                            letter = DXM[int(number)]
                            return letter
                          
                          replaced_text = number_regex.sub(replace_number, text)
                          return replaced_text
                        
                        io = re_numtoletter(x_string,DXM)
                        return '\n' +  io
                        
                xboo = creative_roygbivious_EDP(x)
                x_list = re.split(r'(\d)', str(xboo))
                xbog = [i for i in x_list if i]
                
                
                def replace_letters_with_numbers(text, DXM):
                    # Create a regular expression to match letters.
                    letter_regex = re.compile(r'[a-zA-Z]')
                
                    # Replace the letters in the string with the corresponding numbers.
                    def replace_letter(match):
                        letter = match.group()
                        # Use get() method with a default value of 0 if letter is not found in DXM.
                        number = DXM.get(letter, 0)
                        return str(number)  # Convert number to string before returning.
                
                    replaced_text = letter_regex.sub(replace_letter, text)
                
                    return replaced_text
                
                DXM = {'s':0, 'a':1, 'b':2 , 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}
                scottish = xbog[0]
                rotta = replace_letters_with_numbers(scottish,DXM)
                x_list = re.split(r'(\d)', str(rotta))
                xbog = [i for i in x_list if i]
                x_string = ''.join(xbog)
                numbers = []
                for sublist in x_string:
                    for number in sublist:
                        numbers.append(number)
                
                numbers = numbers[28:143]
                lisst = ''.join(numbers)
                
                bracket_regex = r'\[|\]'
                reedo = re.sub(bracket_regex, '', lisst)
                
                def extract_numbers(string):
                      numbers = []
                      for char in string:
                        if char.isdigit() or char == '.' or char == ',':
                          numbers.append(char)
                      return numbers
                
                numbers = extract_numbers(reedo)
                lisst = ''.join(numbers)
                opp = int(lisst[0])    
                new = np.clip(opp, 1.0, 1.01)
                return new   
     
        def A_way(self, inputs, target): #=================----------......... TALK SHOW ..............---------=================#
           
           
           #.... BLOCK INIT ....................................
                                                     
           dwt1 = self.weight1
           internal_ops = np.mean(self.weight1)
           int_ops = np.cos(internal_ops)/1000
           
           ### ------- SUB INIT ----.............................
           
           cube1 = self.cube1
           
           cube2 = self.cube2
           
           cube3 = self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                dwt2 = self.weight2
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 
                dwt3 = np.array(dwt3)
                w1 = dwt1 
                w4 = dwt2
                w7 = dwt3 
                
                #.................  BLOCK 2 ......................
                L1 = w7
                L2 = w4 
                L3 = w1
                
                for _ in range(1):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)   + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 * target 
                    delta = sintan_deriva(error)                                        
                    dwt1 +=  np.dot(flow1, delta) * int_ops                  
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 * target
                    deltA = sintan_deriva(error2)
                    dwt2 +=  np.dot(flow2, deltA) * int_ops
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 * target
                    delt = sintan_deriva(error3)
                    dwt3 +=  np.dot(flow1, delt) * int_ops
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(1):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            #cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            #cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            #cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home       # HOME BIAS
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r * g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            cube1 +=  np.dot(cube1, A) * int_ops/10
                            cube2 +=  np.dot(cube2, a) * int_ops/10 
    
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home                     
                            O = np.cos(o)
                            d = r * o
                            i = sintan_deriva(d)
                            cube3 +=  np.dot(cube3, i) * int_ops/10
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           #np.savez('scn2.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           #np.savez('Subc2.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           #iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           #joblib.dump(iam, 'Jwts')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
      # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
      # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
        
    
        def AXION(self, T1, T2, T3, T4, T5, T6):
            # Make axon layer
            x1 = T1
            x2 = T2
            x3 = T3
            x4 = T4
            x5 = T5
            x6 = T6
        
            # Range and return
            X = np.array([[-11, -4, -2, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 4, 11]]).T
            y = np.array([-400, -300, -200, -100, -10, -1, -100, -50, -10, -100, -10, -1, 5, 1, 10, 10, 10, 50, 10, 1, 10, 100, 20000, 300, 500])
    
            # Axon block 1
            clf1 = KNeighborsRegressor(n_neighbors=11)
            clf1.fit(X, y)
            p1 = clf1.predict([[x1]])
    
            # Axon block 2
            clf2 = KNeighborsRegressor(n_neighbors=11)
            clf2.fit(X, y)
            p2 = clf2.predict([[x2]])
    
            # Boost block 1
            boostc1 = AdaBoostRegressor(n_estimators=50)
            boostc1.fit(X, y)
            p3 = boostc1.predict([[x3]])
    
            # Boost block 2
            boostc2 = AdaBoostRegressor(n_estimators=50)
            boostc2.fit(X, y)
            p4 = boostc2.predict([[x4]])
    
            p5 = boostc2.predict([[x5]])
    
            p6 = boostc2.predict([[x6]])
    
            # Sub-axon 1
            sub1 = (x1 + x2 + x3) / 3
            p7 = boostc2.predict([[sub1]])
    
            # Sub-axon 2
            sub2 = (x4 + x5 + x6) / 3
            p8 = boostc2.predict([[sub2]])
    
            p9 = boostc2.predict([[1]])
    
            # New target
            new_target = np.array([[p1, p2, p3],
                                   [p4, p5, p6],
                                   [p7, p8, p9]])
    
            # Reshape to 3x3
            new_target = new_target.reshape((3, 3))
    
            return new_target
        
        def parameters(self):
            parameters = {
                #'weights_hidden': self.weights_hidden,
                #'biases_hidden': self.biases_hidden,
                #'biases_output': self.biases_output,
                 'weights_output': np.mean(self.home)}
            return parameters
    
    #===========----------------............................  S Y N A P T I C   G A P    ................ --------- =========== ########    22222222222222
    
    inputs2 = chipperdido(ai)
    
    target2 = y
    nn2 = SCO2(Jwts)
    
    output2 = nn2.forward(inputs2,target2)
    
    N_target2 = Targ(output2, nn2)
    
    oa2 = nn2.PSYCH(output2)
    ai2 = nn2.A_way(output2+oa2, N_target2)
    
    #joblib.dump(nn2, "nn2.joblib")
    
    
    #====================-----------------------.....................   G  A  P   ..............------=============##############     
    load_spirit = np.load('scn33.npz') 
    cube  = np.load('Subc33.npz')                                                             #########################
    #======================================================================================================
    
    class SCO3:             
        def __init__(self,jwts):
            TL = time.time()
            time.sleep(0.1)
            TM = time.time() /                                                                            1e11
            time.sleep(0.000001)
            TR = time.time() /                                                                         1e11
            time.sleep(0.000001)
            LM = time.time() /                                                                       1e11
            time.sleep(0.000001)
            MM = time.time() /                                                                    1e11
            time.sleep(0.000001)
            RM = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BL = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BM = time.time()/                                                                     1e11
            time.sleep(0.000001)
            BR = time.time()/                                                                          1e10
            
            self.home= np.array([
                                        [TL, TM, TR],
                                        [LM, MM, RM],
                                        [BL, BM, BR]])
            self.cymatic = Dyf()
            self.rock = jwts
            ################======--------- SUB-DIVISIONS  
            self.cork = self.rock['cybi1']
            self.kroc = self.rock['cybi2']
            self.ckor = self.rock['cybi3']
            self.home = self.rock['home']  
            self.core1 = self.rock['wt1'] 
            self.core2 = self.rock['wt2']
            self.core3 = self.rock['wt3']
            self.core4 = self.rock['cube1']
            self.core5 = self.rock['cube2']
            self.core6 = self.rock['cube3'] 
            solid = self.core5 ; solid1 = self.core6 ; solid2 = self.core4; solid3 = self.core3 ; solid4 = self.core1 ; solid5 = self.core2
            
            self.cube1 = cube['x'] + solid *0.00001
            self.cube2 = cube['y'] + solid1*0.00001
            self.cube3 = cube['z'] + solid2*0.00001
            self.weight1 = load_spirit['x'] + solid3 *0.00001
            self.weight2 = load_spirit['y'] + solid4 *0.00001
            self.weight3 = load_spirit['z'] + solid5 *0.00001
            print(self.home)
            
        def push(self, item):
             np.append(self.home,item)
             
        def pop(self):
             if self.is_empty():
                 return None
             return self.pop()
    
        def is_empty(self):
            if not self.home in self.home:
              return True
            else:
              return True
         
        def forward(self, inputs, target): # =================----------......... THE MAIN SHOW ..............---------================= #
           
           bias  = self.cymatic
           #----- Positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           bias0 = bias[0][0] ; bias1 = bias[0][1] ; bias2 = bias[0][2]
           bias3 = bias[1][0] ; bias4 = bias[1][1] ; bias5 = bias[1][2]
           bias6 = bias[2][0] ; bias7 = bias[2][1] ; bias8 = bias[2][2]
           self.cybi_1 = np.array([[bias0, bias1, bias2],
                                  [bias3, bias4, bias5],
                                  [bias6, bias7, bias8]]) #+ self.cork/3
          
           #------ negative -''''''''''''''''''----------------------------------------
           bias9 = bias[3][0] ; bias10 = bias[3][1] ; bias11 = bias[3][2]
           bias12 = bias[4][0] ; bias13 = bias[4][1] ; bias14 = bias[4][2]
           bias15 = bias[5][0] ; bias16 = bias[5][1] ; bias17 = bias[5][2]
           self.cybi_2 =  np.array([[bias9, bias10, bias11],
                                  [bias12, bias13, bias14],
                                  [bias15, bias16, bias17]])  #+ self.kroc/3
           #------- positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           
           bias18 = bias[6][0] ; bias19 = bias[6][1] ; bias20 = bias[6][2]
           bias21 = bias[7][0] ; bias22 = bias[7][1] ; bias23 = bias[7][2]
           bias24 = bias[8][0] ; bias25 = bias[8][1] ; bias26 = bias[8][2]
           self.cybi_3 =  np.array([[bias18, bias19, bias20],
                                  [bias21, bias22, bias23],
                                  [bias24, bias25, bias26]]) # + self.ckor/3
           
           #.... BLOCK INIT ....................................
           top =  self.cybi_1                                           
           dwt1 = self.weight1 + top
           internal_ops = np.mean(top-self.weight1)
           int_ops = np.cos(internal_ops)*0.00001
           
           
           ### ------- SUB INIT ----.............................
           Xinp = self.cybi_1[:, :, np.newaxis]
           Xi = Xinp.reshape((9))
           cube1 = Xi -self.cube1
           
           Xip = self.cybi_2[:, :, np.newaxis]
           X = Xip.reshape((9))
           cube2 = X + self.cube2
           
           Xp = self.cybi_3[:, :, np.newaxis]
           x = Xp.reshape((9))
           cube3 = x - self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                t = np.array(self.cybi_2)
                dwt2 = self.weight2   - t
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 +  self.cybi_3
                dwt3 = np.array(dwt3)
                # COLUNM DOWN GROUPING AND SHUFFLED
                w1 = dwt1[:,0] ; w2 = dwt1[:,1] ; w3 = dwt1[:,2] 
                w4 = dwt2[:,0] ; w5 = dwt2[:,1] ; w6 = dwt2[:,2] 
                w7 = dwt3[:,0] ; w8 = dwt3[:,1] ; w9 = dwt3[:,2] # At this point I have combined "movements" with all weights respectively based on universal law within a computer 
                
                #.................  BLOCK 2 ......................
                #  three arrays into a single array
                L1 = np.concatenate((w1,w4,w7)).reshape(3, 3) #-%
                L2 = np.concatenate((w2,w5,w8)).reshape(3, 3) 
                L3 = np.concatenate((w3,w6,w9)).reshape(3, 3) 
                
                
                for _ in range(50):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)     + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 - target 
                    delta = sintan_deriva(error)                                        
                    self.weight1 +=  np.dot(flow1, delta) * int_ops                  
                    self.push(self.home)
                    
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 - target
                    deltA = sintan_deriva(error2)
                    self.weight2 +=  np.dot(flow2, deltA) * int_ops
         
                    self.pop()
                    
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 - target
                    delt = sintan_deriva(error3)
                    self.weight3 +=  np.dot(flow1, delt) * int_ops
                    self.home-= int_ops * delt               
                    
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(34):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home  
                            #g = c_dropout(g)       
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r - g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            self.cube1 +=  np.dot(cube1, A) * int_ops
                            self.cube2 +=  np.dot(cube2, a) * int_ops 
                           # xc = chip(0.001)
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home    
                            #o = c_dropout(o)
                            O = np.cos(o)
                            d = r - o
                            i = sintan_deriva(d)
                            self.cube3 +=  np.dot(cube3, i) * int_ops
                            
                           # self.home-=  xc * int_ops
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           np.savez('scn33.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           np.savez('Subc33.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           joblib.dump(iam, 'Jwts4')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
        # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
        # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
    
        def PSYCH(self,x):
                x=x
                def creative_roygbivious_EDP(x):
                    #x= '\033[0m' RESET COLOR or end of color reach
                        # Split the number into a list of strings
                        x_list = re.split(r'(\d)', str(x))
                        
                        a = '\033[31m' #red
                        b = '\033[30;1;33m' 
                        c = '\033[36m'
                        d = '\033[32m' # GREEN
                        e = '\033[30;1;29m'
                        f = '\033[35m'
                        g = '\033[30;1;2m' 
                        
                        
                        # Remove any empty strings from the list
                        x_list = [i for i in x_list if i]
                        for _ in range(5):
                           xs1 = g +''.join(x_list)
                           print(xs1*4)
                           xs2 = b +''.join(x_list)
                           print(xs2*10)
                           xs3 = c +''.join(x_list)
                           print(xs3*10)
                           xs4 = d +''.join(x_list)
                           print(xs4*10)
                           xs5 = e +''.join(x_list)
                           print(xs5*10)
                           xs6 = f +''.join(x_list)         
                           print(xs6*10)
                           xs7 = a +''.join(x_list)
                           print(xs7*10)
                           x_list = xs7 +xs4 + xs3 + xs6 +xs2 +xs1 +xs5
                           x_list
                           x_list = xs7
                        x_string = ''.join(x_list)
                        
                        DXM = {0:'s', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i'}
                        
                        def re_numtoletter(text, DXM):
                        
                          # Create a regular expression to match numbers.
                          number_regex = re.compile(r'\d')
                        
                          # Replace the numbers in the string with the corresponding letters.
                          def replace_number(match):
                            number = match.group()
                            letter = DXM[int(number)]
                            return letter
                          
                          replaced_text = number_regex.sub(replace_number, text)
                          return replaced_text
                        
                        io = re_numtoletter(x_string,DXM)
                        return '\n' +  io
                        
                xboo = creative_roygbivious_EDP(x)
                x_list = re.split(r'(\d)', str(xboo))
                xbog = [i for i in x_list if i]
                
                
                def replace_letters_with_numbers(text, DXM):
                    # Create a regular expression to match letters.
                    letter_regex = re.compile(r'[a-zA-Z]')
                
                    # Replace the letters in the string with the corresponding numbers.
                    def replace_letter(match):
                        letter = match.group()
                        # Use get() method with a default value of 0 if letter is not found in DXM.
                        number = DXM.get(letter, 0)
                        return str(number)  # Convert number to string before returning.
                
                    replaced_text = letter_regex.sub(replace_letter, text)
                
                    return replaced_text
                
                DXM = {'s':0, 'a':1, 'b':2 , 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}
                scottish = xbog[0]
                rotta = replace_letters_with_numbers(scottish,DXM)
                x_list = re.split(r'(\d)', str(rotta))
                xbog = [i for i in x_list if i]
                x_string = ''.join(xbog)
                numbers = []
                for sublist in x_string:
                    for number in sublist:
                        numbers.append(number)
                
                numbers = numbers[28:143]
                lisst = ''.join(numbers)
                
                bracket_regex = r'\[|\]'
                reedo = re.sub(bracket_regex, '', lisst)
                
                def extract_numbers(string):
                      numbers = []
                      for char in string:
                        if char.isdigit() or char == '.' or char == ',':
                          numbers.append(char)
                      return numbers
                
                numbers = extract_numbers(reedo)
                lisst = ''.join(numbers)
                opp = int(lisst[0])    
                new = np.clip(opp, 1.0, 1.01)
                return new   
     
        def A_way(self, inputs, target): #=================----------......... TALK SHOW ..............---------=================#
           
           
           #.... BLOCK INIT ....................................
                                                     
           dwt1 = self.weight1
           internal_ops = np.mean(self.weight1)
           int_ops = np.cos(internal_ops)/1000
           
           ### ------- SUB INIT ----.............................
           
           cube1 = self.cube1
           
           cube2 = self.cube2
           
           cube3 = self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                dwt2 = self.weight2
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 
                dwt3 = np.array(dwt3)
                w1 = dwt1 
                w4 = dwt2
                w7 = dwt3 
                
                #.................  BLOCK 2 ......................
                L1 = w7
                L2 = w4 
                L3 = w1
                
                for _ in range(1):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)   + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 * target 
                    delta = sintan_deriva(error)                                        
                    dwt1 +=  np.dot(flow1, delta) * int_ops                  
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 * target
                    deltA = sintan_deriva(error2)
                    dwt2 +=  np.dot(flow2, deltA) * int_ops
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 * target
                    delt = sintan_deriva(error3)
                    dwt3 +=  np.dot(flow1, delt) * int_ops
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(1):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            #cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            #cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            #cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home       # HOME BIAS
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r * g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            cube1 +=  np.dot(cube1, A) * int_ops/10
                            cube2 +=  np.dot(cube2, a) * int_ops/10 
    
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home                     
                            O = np.cos(o)
                            d = r * o
                            i = sintan_deriva(d)
                            cube3 +=  np.dot(cube3, i) * int_ops/10
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           #np.savez('scn2.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           #np.savez('Subc2.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           #iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           #joblib.dump(iam, 'Jwts')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
      # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
      # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
        
    
        def AXION(self, T1, T2, T3, T4, T5, T6):
            # Make axon layer
            x1 = T1
            x2 = T2
            x3 = T3
            x4 = T4
            x5 = T5
            x6 = T6
        
            # Range and return
            X = np.array([[-11, -4, -2, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 4, 11]]).T
            y = np.array([-400, -300, -200, -100, -10, -1, -100, -50, -10, -100, -10, -1, 5, 1, 10, 10, 10, 50, 10, 1, 10, 100, 20000, 300, 500])
    
            # Axon block 1
            clf1 = KNeighborsRegressor(n_neighbors=11)
            clf1.fit(X, y)
            p1 = clf1.predict([[x1]])
    
            # Axon block 2
            clf2 = KNeighborsRegressor(n_neighbors=11)
            clf2.fit(X, y)
            p2 = clf2.predict([[x2]])
    
            # Boost block 1
            boostc1 = AdaBoostRegressor(n_estimators=50)
            boostc1.fit(X, y)
            p3 = boostc1.predict([[x3]])
    
            # Boost block 2
            boostc2 = AdaBoostRegressor(n_estimators=50)
            boostc2.fit(X, y)
            p4 = boostc2.predict([[x4]])
    
            p5 = boostc2.predict([[x5]])
    
            p6 = boostc2.predict([[x6]])
    
            # Sub-axon 1
            sub1 = (x1 + x2 + x3) / 3
            p7 = boostc2.predict([[sub1]])
    
            # Sub-axon 2
            sub2 = (x4 + x5 + x6) / 3
            p8 = boostc2.predict([[sub2]])
    
            p9 = boostc2.predict([[1]])
    
            # New target
            new_target = np.array([[p1, p2, p3],
                                   [p4, p5, p6],
                                   [p7, p8, p9]])
    
            # Reshape to 3x3
            new_target = new_target.reshape((3, 3))
    
            return new_target
        
        def parameters(self):
            parameters = {
                #'weights_hidden': self.weights_hidden,
                #'biases_hidden': self.biases_hidden,
                #'biases_output': self.biases_output,
                 'weights_output': np.mean(self.home)}
            return parameters
    
    #===========----------------............................  S Y N A P T I C   G A P    ................ --------- =========== ########    1 1 1  1  1 1 1 
    
    inputs3 = chipperdido(ai2)
    target3 = y
    
    nn3 = SCO3(Jwts)
    
    output3 = nn3.forward(inputs3,target3)
    
    N_target3 = Targ(output3, nn3)
    
    oa3 = nn3.PSYCH(output3)
    ai3 = nn3.A_way(output3+oa3, N_target3)
    
    
    
    #====================-----------------------.....................   G  A  P   ..............------=============##############     
    load_spirit = np.load('scn34.npz') 
    cube  = np.load('Subc34.npz')                                                             #########################
    #======================================================================================================
    
    class SCO4:             
        def __init__(self,jwts):
            TL = time.time()
            time.sleep(0.1)
            TM = time.time() /                                                                            1e11
            time.sleep(0.000001)
            TR = time.time() /                                                                         1e11
            time.sleep(0.000001)
            LM = time.time() /                                                                       1e11
            time.sleep(0.000001)
            MM = time.time() /                                                                    1e11
            time.sleep(0.000001)
            RM = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BL = time.time() /                                                                  1e11
            time.sleep(0.000001)
            BM = time.time()/                                                                     1e11
            time.sleep(0.000001)
            BR = time.time()/                                                                          1e10
            
            self.home= np.array([
                                        [TL, TM, TR],
                                        [LM, MM, RM],
                                        [BL, BM, BR]])
            self.cymatic = Dyf()
            self.rock = jwts
            ################======--------- SUB-DIVISIONS  
            self.cork = self.rock['cybi1']
            self.kroc = self.rock['cybi2']
            self.ckor = self.rock['cybi3']
            self.home = self.rock['home']  
            self.core1 = self.rock['wt1'] 
            self.core2 = self.rock['wt2']
            self.core3 = self.rock['wt3']
            self.core4 = self.rock['cube1']
            self.core5 = self.rock['cube2']
            self.core6 = self.rock['cube3'] 
            solid = self.core5 ; solid1 = self.core6 ; solid2 = self.core4; solid3 = self.core3 ; solid4 = self.core1 ; solid5 = self.core2
            
            self.cube1 = cube['x'] + solid *0.00001
            self.cube2 = cube['y'] + solid1*0.00001
            self.cube3 = cube['z'] + solid2*0.00001
            self.weight1 = load_spirit['x'] + solid3 *0.00001
            self.weight2 = load_spirit['y'] + solid4 *0.00001
            self.weight3 = load_spirit['z'] + solid5 *0.00001
            print(self.home)
            
        def push(self, item):
             np.append(self.home,item)
             
        def pop(self):
             if self.is_empty():
                 return None
             return self.pop()
    
        def is_empty(self):
            if not self.home in self.home:
              return True
            else:
              return True
         
        def forward(self, inputs, target): # =================----------......... THE MAIN SHOW ..............---------================= #
           
           bias  = self.cymatic
           #----- Positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           bias0 = bias[0][0] ; bias1 = bias[0][1] ; bias2 = bias[0][2]
           bias3 = bias[1][0] ; bias4 = bias[1][1] ; bias5 = bias[1][2]
           bias6 = bias[2][0] ; bias7 = bias[2][1] ; bias8 = bias[2][2]
           self.cybi_1 = np.array([[bias0, bias1, bias2],
                                  [bias3, bias4, bias5],
                                  [bias6, bias7, bias8]]) #+ self.cork/3
          
           #------ negative -''''''''''''''''''----------------------------------------
           bias9 = bias[3][0] ; bias10 = bias[3][1] ; bias11 = bias[3][2]
           bias12 = bias[4][0] ; bias13 = bias[4][1] ; bias14 = bias[4][2]
           bias15 = bias[5][0] ; bias16 = bias[5][1] ; bias17 = bias[5][2]
           self.cybi_2 =  np.array([[bias9, bias10, bias11],
                                  [bias12, bias13, bias14],
                                  [bias15, bias16, bias17]])  #+ self.kroc/3
           #------- positive +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           
           bias18 = bias[6][0] ; bias19 = bias[6][1] ; bias20 = bias[6][2]
           bias21 = bias[7][0] ; bias22 = bias[7][1] ; bias23 = bias[7][2]
           bias24 = bias[8][0] ; bias25 = bias[8][1] ; bias26 = bias[8][2]
           self.cybi_3 =  np.array([[bias18, bias19, bias20],
                                  [bias21, bias22, bias23],
                                  [bias24, bias25, bias26]]) # + self.ckor/3
           
           #.... BLOCK INIT ....................................
           top =  self.cybi_1                                           
           dwt1 = self.weight1 + top
           internal_ops = np.mean(top-self.weight1)
           int_ops = np.cos(internal_ops)*0.00001
           
           
           ### ------- SUB INIT ----.............................
           Xinp = self.cybi_1[:, :, np.newaxis]
           Xi = Xinp.reshape((9))
           cube1 = Xi -self.cube1
           
           Xip = self.cybi_2[:, :, np.newaxis]
           X = Xip.reshape((9))
           cube2 = X + self.cube2
           
           Xp = self.cybi_3[:, :, np.newaxis]
           x = Xp.reshape((9))
           cube3 = x - self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                t = np.array(self.cybi_2)
                dwt2 = self.weight2   - t
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 +  self.cybi_3
                dwt3 = np.array(dwt3)
                # COLUNM DOWN GROUPING AND SHUFFLED
                w1 = dwt1[:,0] ; w2 = dwt1[:,1] ; w3 = dwt1[:,2] 
                w4 = dwt2[:,0] ; w5 = dwt2[:,1] ; w6 = dwt2[:,2] 
                w7 = dwt3[:,0] ; w8 = dwt3[:,1] ; w9 = dwt3[:,2] # At this point I have combined "movements" with all weights respectively based on universal law within a computer 
                
                #.................  BLOCK 2 ......................
                #  three arrays into a single array
                L1 = np.concatenate((w1,w4,w7)).reshape(3, 3) #-%
                L2 = np.concatenate((w2,w5,w8)).reshape(3, 3) 
                L3 = np.concatenate((w3,w6,w9)).reshape(3, 3) 
                
                
                for _ in range(50):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)     + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 - target 
                    delta = sintan_deriva(error)                                        
                    self.weight1 +=  np.dot(flow1, delta) * int_ops                  
                    self.push(self.home)
                    
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 - target
                    deltA = sintan_deriva(error2)
                    self.weight2 +=  np.dot(flow2, deltA) * int_ops
         
                    self.pop()
                    
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 - target
                    delt = sintan_deriva(error3)
                    self.weight3 +=  np.dot(flow1, delt) * int_ops
                    self.home-= int_ops * delt               
                    
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(34):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home  
                            #g = c_dropout(g)       
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r - g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            self.cube1 +=  np.dot(cube1, A) * int_ops
                            self.cube2 +=  np.dot(cube2, a) * int_ops 
                           # xc = chip(0.001)
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home    
                            #o = c_dropout(o)
                            O = np.cos(o)
                            d = r - o
                            i = sintan_deriva(d)
                            self.cube3 +=  np.dot(cube3, i) * int_ops
                            
                           # self.home-=  xc * int_ops
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           np.savez('scn34.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           np.savez('Subc34.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           joblib.dump(iam, 'Jwts4')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
        # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
        # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
    
        def PSYCH(self,x):
                x=x
                def creative_roygbivious_EDP(x):
                    #x= '\033[0m' RESET COLOR or end of color reach
                        # Split the number into a list of strings
                        x_list = re.split(r'(\d)', str(x))
                        
                        a = '\033[31m' #red
                        b = '\033[30;1;33m' 
                        c = '\033[36m'
                        d = '\033[32m' # GREEN
                        e = '\033[30;1;29m'
                        f = '\033[35m'
                        g = '\033[30;1;2m' 
                        
                        
                        # Remove any empty strings from the list
                        x_list = [i for i in x_list if i]
                        for _ in range(5):
                           xs1 = g +''.join(x_list)
                           print(xs1*4)
                           xs2 = b +''.join(x_list)
                           print(xs2*10)
                           xs3 = c +''.join(x_list)
                           print(xs3*10)
                           xs4 = d +''.join(x_list)
                           print(xs4*10)
                           xs5 = e +''.join(x_list)
                           print(xs5*10)
                           xs6 = f +''.join(x_list)         
                           print(xs6*10)
                           xs7 = a +''.join(x_list)
                           print(xs7*10)
                           x_list = xs7 +xs4 + xs3 + xs6 +xs2 +xs1 +xs5
                           x_list
                           x_list = xs7
                        x_string = ''.join(x_list)
                        
                        DXM = {0:'s', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i'}
                        
                        def re_numtoletter(text, DXM):
                        
                          # Create a regular expression to match numbers.
                          number_regex = re.compile(r'\d')
                        
                          # Replace the numbers in the string with the corresponding letters.
                          def replace_number(match):
                            number = match.group()
                            letter = DXM[int(number)]
                            return letter
                          
                          replaced_text = number_regex.sub(replace_number, text)
                          return replaced_text
                        
                        io = re_numtoletter(x_string,DXM)
                        return '\n' +  io
                        
                xboo = creative_roygbivious_EDP(x)
                x_list = re.split(r'(\d)', str(xboo))
                xbog = [i for i in x_list if i]
                
                
                def replace_letters_with_numbers(text, DXM):
                    # Create a regular expression to match letters.
                    letter_regex = re.compile(r'[a-zA-Z]')
                
                    # Replace the letters in the string with the corresponding numbers.
                    def replace_letter(match):
                        letter = match.group()
                        # Use get() method with a default value of 0 if letter is not found in DXM.
                        number = DXM.get(letter, 0)
                        return str(number)  # Convert number to string before returning.
                
                    replaced_text = letter_regex.sub(replace_letter, text)
                
                    return replaced_text
                
                DXM = {'s':0, 'a':1, 'b':2 , 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}
                scottish = xbog[0]
                rotta = replace_letters_with_numbers(scottish,DXM)
                x_list = re.split(r'(\d)', str(rotta))
                xbog = [i for i in x_list if i]
                x_string = ''.join(xbog)
                numbers = []
                for sublist in x_string:
                    for number in sublist:
                        numbers.append(number)
                
                numbers = numbers[28:143]
                lisst = ''.join(numbers)
                
                bracket_regex = r'\[|\]'
                reedo = re.sub(bracket_regex, '', lisst)
                
                def extract_numbers(string):
                      numbers = []
                      for char in string:
                        if char.isdigit() or char == '.' or char == ',':
                          numbers.append(char)
                      return numbers
                
                numbers = extract_numbers(reedo)
                lisst = ''.join(numbers)
                opp = int(lisst[0])    
                new = np.clip(opp, 1.0, 1.01)
                return new   
     
        def A_way(self, inputs, target): #=================----------......... TALK SHOW ..............---------=================#
           
           
           #.... BLOCK INIT ....................................
                                                     
           dwt1 = self.weight1
           internal_ops = np.mean(self.weight1)
           int_ops = np.cos(internal_ops)/1000
           
           ### ------- SUB INIT ----.............................
           
           cube1 = self.cube1
           
           cube2 = self.cube2
           
           cube3 = self.cube3   # MOVEMENT PATTERN SMALL CUBES + - + BIG CUBES - + - 
           self.push(self.home)                         
    
           for epoch in range(1):   
               
                #...............  BLOCK 1 ......................
                dwt2 = self.weight2
                dwt2 = np.array(dwt2)
                
                dwt3 = self.weight3 
                dwt3 = np.array(dwt3)
                w1 = dwt1 
                w4 = dwt2
                w7 = dwt3 
                
                #.................  BLOCK 2 ......................
                L1 = w7
                L2 = w4 
                L3 = w1
                
                for _ in range(1):
                    
                    #.......  BLOCK 3 ......................                                                              # SEE b , d and p are all the same letter just looked at differently...
                    flow1 =  np.dot(L1,inputs)   + self.home                                          # MATRIX MULT. 
                    flow1 = c_dropout(flow1)
                    output_1 = np.cos(flow1)                 # ACTIVATION FUNC
                    error = output_1 * target 
                    delta = sintan_deriva(error)                                        
                    dwt1 +=  np.dot(flow1, delta) * int_ops                  
                 #--------
                 
                    outer= output_1/3
                    
                    #.......  BLOCK 4 ......................
                    flow2 = np.dot(flow1,L2) + self.home   # *0                    #reset 
                    #flow2 = c_dropout(flow2)              # DROPOUT PATTERN 50% zeross 
                    output_2 = np.cos(flow2)
                    error2 = output_2 * target
                    deltA = sintan_deriva(error2)
                    dwt2 +=  np.dot(flow2, deltA) * int_ops
                 #--------
                 
                    space = output_2/2
                    
                    #.......  BLOCK 5 ......................
                    flow3 =  np.dot(flow2,L3) + self.home # *0                #reset 
                    #flow3 =  c_dropout(flow3)
                    output_3 = np.cos(flow3)
                    error3 = output_3 * target
                    delt = sintan_deriva(error3)
                    dwt3 +=  np.dot(flow1, delt) * int_ops
                    mannn = (outer+space+output_3)/3
                    mannn = mannn
    
                    for _ in range(1):             
                                                              # SEE b , d and p are all the same letter just looked at differently...
                            b1 = output_1[:, :, np.newaxis]
                            b1 = b1.reshape((9))     
                            cu1 =                 np.dot(b1, cube1)  # *0 #reset
                            #cu1 = c_dropout(cu1)
                        #====-----------------------------------  6
                            
                            d1 = output_2[:, :, np.newaxis]
                            d1 = d1.reshape((9))
                            cu2 =                np.dot(d1, cube2)   # *0  #reset
                            #cu2 = c_dropout(cu2)
                        #====-----------------------------------  7
                        
                            p1 = output_3[:, :, np.newaxis]
                            p1 = p1.reshape((9))
                            cu3 =               np.dot(p1, cube3)   # *0    #reset
                            #cu3 = c_dropout(cu3)
                        #====-----------------------------------  8    6 - 8 is reshape and matrix math 
                           
                            B8 = self.home[:, :, np.newaxis]
                            home = B8.reshape((9))
                            g = np.dot(cu1,cu2)  + home       # HOME BIAS
                            G = np.cos(g)
                            
                            rr = error3[:, :, np.newaxis]                              # tail where back-p happens for big cubes  
                            r = rr.reshape((9))
                            err  = r * g
                            
                            A = sintan_deriva(err)
                            a = sintan_deriva(A)
                            
                            cube1 +=  np.dot(cube1, A) * int_ops/10
                            cube2 +=  np.dot(cube2, a) * int_ops/10 
    
                        #====-----------------------------------  9    
                               
                            o = np.dot(g, cu3)  + home                     
                            O = np.cos(o)
                            d = r * o
                            i = sintan_deriva(d)
                            cube3 +=  np.dot(cube3, i) * int_ops/10
                            D = O+G.                                                                                                             T   # A MIRROR ACT I ON
                        #====-----------------------------------  10
                            pic = clip(D)
                            self.E = np.dot(mannn, pic)   
                            #self.push(self.home+0.04) 
                            
           #np.savez('scn2.npz', x=self.weight1, y=self.weight2, z=self.weight3)
           #np.savez('Subc2.npz', x=self.cube1, y=self.cube2, z=self.cube3)
           #iam = {'home':self.home, 'wt1':self.weight1,'wt2':self.weight2,'wt3':self.weight3,'cube1':self.cube1,'cube2':self.cube2,'cube3':self.cube3,'cybi1': self.cybi_1, 'cybi2':self.cybi_2, 'cybi3': self.cybi_3}
           #joblib.dump(iam, 'Jwts')  
                
        #### ---------- MODULAR INFOMETRIC BIOTELEMETRY  ------------#####
       
        # 3X3 PROCESS  ------     
           PT = np.sum(flow1 - flow2)
           pt = np.sum(flow2 + flow3)
           percent = PT / pt * 100
           #print('\n ============= WT % peRCENT [3X3] :',percent) #'\n ============= [3X3] ORIGINAL OUTPUT : \n',output_3,'\n')
      # 9X9X9 PROCESS  ------
           PT = np.sum( G - O )
           pt = PT
           percent =  pt * 100
           #print('-------------  WT % PErcent [9X9X9]:',percent)   
      # OUTPUT PARAMETERS
           #print( '\n ============= FIRST 9X9X9 MATH GUTS (CLIPPED) :\n',clip(o), '\n\n ============= FINAL OUTPUT [3X3] :\n', self.E, '\n ============= LEARNING RATE : \n', int_ops, )                  
                   
           return  self.E
        
    
        def AXION(self, T1, T2, T3, T4, T5, T6):
            # Make axon layer
            x1 = T1
            x2 = T2
            x3 = T3
            x4 = T4
            x5 = T5
            x6 = T6
        
            # Range and return
            X = np.array([[-11, -4, -2, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 4, 11]]).T
            y = np.array([-400, -300, -200, -100, -10, -1, -100, -50, -10, -100, -10, -1, 5, 1, 10, 10, 10, 50, 10, 1, 10, 100, 20000, 300, 500])
    
            # Axon block 1
            clf1 = KNeighborsRegressor(n_neighbors=11)
            clf1.fit(X, y)
            p1 = clf1.predict([[x1]])
    
            # Axon block 2
            clf2 = KNeighborsRegressor(n_neighbors=11)
            clf2.fit(X, y)
            p2 = clf2.predict([[x2]])
    
            # Boost block 1
            boostc1 = AdaBoostRegressor(n_estimators=50)
            boostc1.fit(X, y)
            p3 = boostc1.predict([[x3]])
    
            # Boost block 2
            boostc2 = AdaBoostRegressor(n_estimators=50)
            boostc2.fit(X, y)
            p4 = boostc2.predict([[x4]])
    
            p5 = boostc2.predict([[x5]])
    
            p6 = boostc2.predict([[x6]])
    
            # Sub-axon 1
            sub1 = (x1 + x2 + x3) / 3
            p7 = boostc2.predict([[sub1]])
    
            # Sub-axon 2
            sub2 = (x4 + x5 + x6) / 3
            p8 = boostc2.predict([[sub2]])
    
            p9 = boostc2.predict([[1]])
    
            # New target
            new_target = np.array([[p1, p2, p3],
                                   [p4, p5, p6],
                                   [p7, p8, p9]])
    
            # Reshape to 3x3
            new_target = new_target.reshape((3, 3))
    
            return new_target
        
        def parameters(self):
            parameters = {
                #'weights_hidden': self.weights_hidden,
                #'biases_hidden': self.biases_hidden,
                #'biases_output': self.biases_output,
                 'weights_output': np.mean(self.home)}
            return parameters
    
    #===========----------------............................  S Y N A P T I C   G A P    ................ --------- =========== ########    1 1 1  1  1 1 1 
    
    inputs4 = chipperdido(ai3)
    target4 = y
    
    nn4 = SCO4(Jwts)
    
    output4 = nn4.forward(inputs4,target4)
    
    N_target4 = Targ(output4, nn4)
    
    oa4 = nn4.PSYCH(output4)
    ai4 = nn4.A_way(output4+oa4, N_target4)
    
    
    return ai4
    
    
    
#print(depolorize([[-0.15222846,  0.88082343,  0.91212123], [-0.91304799,  0.98161611, -0.98122095],  [-0.32155498, -0.64036311, -0.75064098]]))  
    
     
    
    
    
    
    
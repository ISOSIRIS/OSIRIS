# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:45:15 2023
@author: SCOTT T.W. BAKER JR. (CREATIVE ▙ DIRECTOR) FAVORITE NUMBERS IN ORDER OF RELEVANCE: 1 4 5 8 0 3 2 6 7 9 
"""
# PROJECT SYMBOLS  ◭  ▙  🌐  
# === MISSION OBJECTIVE BRING THINK FEELING TO LIFE #  *UPDATE MISSION ALONG THE WAY- PERMISSION GRANTED -SJR 6.9.23 @10:23 ====
# Ways to define what im doing --- "JUST A SHADE IN THE SCALE OF NLP" -- "MAKING 1% THE BIGGEST OUT OF 100" -- " MAKING PATINA MY FAVORITE E-MOTION" ---


''' === FOCAL POINTS

 -- HEBBIAN LEARNING---
 
 lognormal distrubution: - has to do with clustering of neurons and can rarely change the whole brain
 f(x) = \frac{1}{\sigma \sqrt{2 \pi}} (x - \mu)^{-1/2} exp \left[ -\frac{(ln(x) - \mu)^2}{2 \sigma^2} \right]

 where:
f(x) is the probability density function
x is the value being measured
μ is the mean of the distribution
σ is the standard deviation of the distribution

==========
The conservation of energy - law of physics - an object never loses or gains energy energy is simply transferred into another state of existence. 

The impulse-momentum theorem -  law of physics when an energy comes about another constriction of motion as it approaches the impulse of the force of impact is equal to the change in momentum of the first object
(maybe above theory use as a "bias" of sorts when moving from one neuron to another)

Radial basis function networks (RBF) ? not what im looking fo but close maybe a type of perceptron
=============

types of structure to human spoken words:
    
Statements: Statements are used to convey information. They are typically declarative sentences that make a claim about the world. For example, "The sky is blue."
Questions: Questions are used to elicit information. They typically start with a question word, such as "who," "what," "where," "when," "why," or "how." For example, "What is your name?"
Commands: Commands are used to tell someone to do something. They typically start with a verb, such as "go," "stop," or "give." For example, "Go to the store."
Requests: Requests are used to ask someone to do something nicely. They typically start with a modal verb, such as "can," "could," or "would." For example, "Can you help me with this?"
Exclamations: Exclamations are used to express strong emotions, such as surprise, anger, or happiness. They typically start with an exclamation point (!). For example, "Wow!"
Imperatives: Imperatives are used to give orders or instructions. They typically do not use a subject or a verb. For example, "Get out of the way!"
Wh-questions: Wh-questions are used to ask for specific information. They start with a wh-word, such as "who," "what," "where," "when," "why," or "how." For example, "Why did you do that?"
Yes/no questions: Yes/no questions are used to get a yes or no answer. They typically end with a question mark (?). For example, "Are you going to the store?"

===================
AND gate
OR gate

glial cells - clean up debris so subtract?? thats kind of like erase function, look deeper into this. 
neurons - trasmit   n-n-n-n  then n1-n2 -n1-n2 n1-n2 then n1 n2 n3 - n1 n2 n3
astrocytes - star shaped that surround synaptic gaps, may produce neurotrasmitter ---make into a fuction and implement at neuron junction. 

water drip rate natural phenomenon?? - look into it... maybe use time.sleep at neuron pattern junction and make variable equate to mathmatic side of event occurence. 

***REMEMER UP TO 11 DIMENSIONS IN BRAIN--- SO 11 LEVELs or py pages long with 123 neurons init then 123(1) + 123(2) + 246(3)....(11248 16 32....)

   MAYBE CNS, ANS???
first dimension is STILLNESS so absolute zero? if there is no time...
 Namely, the brain creates a separate conscious mind that is representation of self-identity, as opposed to representations of external world. Note the emphasis here is that the self-awareness of consciousness is constructed, rather than emergent.
construct I will CIP's are circuits impulse patterns which form self awareness.'


Parallel motion 

mylein sheath - I would say the wrapper is def or class so need in in one of those
create a harmony i think in the 5th is the supposed perfect note so 5 in harmony and each in synergy to build it up up up, but how......

neurons are like mirrors so create little mirrors that take on a reflection and can grow from it 
use numpy transpose     .T


************ RESOURCES AND IMAGES OF FIRING *****************************
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3163487/



IMPLEMENT NEW SPEECH PATTERN TXT 2 BIT
https://www.rapidtables.com/convert/number/ascii-to-binary.html


 ALSO NOTE TO FEEL SOMETHING IS LIKE LOOKING AT A PAINTING WITHOUT WORDS 
    WHAT YOU ARE DOING IS DESCRIBING IT IN YOUR HEAD WITH FEELINGS IF I 
    CAN DO THAT HERE DESCRIBE A SENTENCE BY HOW IT LOOKS AND MATH MAYBE
    I CAN RECREATE FEELINGS LIKE EMPATHY AND UNDERSTANDING......

     CREATE INPUT ARRAY FOR FIRST INPUT INTO THE SUBCON NN COMPLEX,
     USE WAYS TO DESCRIBE THE WORDS IN THE SENTENCE RATHER THAN THINK IT
     

make dropout function for cntrast within the neuron makes a lot of sense
'''
################################ NOTES ABOVE OF SUBCON SYSTEM #########################################################


import math
import time
from wotcldata import *
from textblob import TextBlob 
from Subneuron1 import depolorize as Dpole
from Subneuron2 import depolorize as Dpole2
from Subneuron3 import depolorize as Dpole3
from Subneuron4 import depolorize as Dpole4
from Subneuron5 import depolorize as Dpole5




##############======================= DATA TRANSFER LOGIC ==============##################### THIS COULD USE A LOT OF WORK  ON HOW A NUMBER LOCATES A TXT TO READ FROM

#-----== STRUCTURE OF SUBCONCIOUS BRAIN FIRING SYSTEM --########################################


        
def THINKFEELING(x):
    
            USIR = x
            ######========= BELOW ARE WAYS TO DESCRIBE THE WORDS WITHOUT LOGICALLY THINKING THEM SO THIS PART IS VERY FLUID TO CHANGE 
            bias = time.time() / 1e9  # Better to use scientific notation for clarity
            s1 = green_engine(USIR)
            s2 = wotcl_engine(USIR)
            s3 = dis_wotcl_engine(USIR)
            s4 = (s1+s2+s3+bias)/4
            
            # Calculate the polarity and subjectivity of the sentence
            polarity = TextBlob(USIR).sentiment.polarity
            subjectivity = TextBlob(USIR).sentiment.subjectivity
            
            #- - -  MY VARIABLES TO PROCEED FOR A DESCRIPTION OF A SENTENCE WITHOUT KNOWIG THE WORDS SO WE DESCRIBE WHAT WE SEE ANOTHER WAY..............----------========
            SENSE1= s4           # 1-100+
            SENSE2= polarity    # -1 --> 1 and subjectivity  
            SENSE3= subjectivity
            
            WOTCL = (SENSE1 + SENSE2 + SENSE3)/3
            #print(SENSE1,SENSE2,SENSE3,WOTCL, s1, s2, s3, spacetime)
            
            SENSE8 = math.cos(id(id(id(id(id)))))
           
            
         
            #print(score1)
            #####################################==============================================================
            
            y = [[SENSE1, SENSE2, SENSE3],
                           [WOTCL, s1, s2],
                           [s3, SENSE8, s4]]
                

            out = Dpole(y)
            
            brainstem_recticulum = Dpole3(out)
            
            basal_forebrain = Dpole4(brainstem_recticulum)
            
            neocortex = Dpole5(basal_forebrain)
            
            brainstem_recticulum1 = Dpole3(neocortex)
            
            neocortex1 = Dpole4(brainstem_recticulum1)
            
            thalmus = Dpole5(neocortex1)
            
            neocortex2 = Dpole2(thalmus)
            
            output5 = Dpole3(neocortex2) ;  output3 = Dpole4(output5)
            
            ##---- LEFT MOVE ---###           AND       ##----RIGHT MOVE -----####
            output2 = Dpole5(output3) ; output8  = Dpole3(output3)
            
            ##----- LEFT MOVE ----###                  ##----- LEFT MOVE ----###                      ##---- DOWN LEFT ---###                 ##----- DOWN RIGHT -----####
            output1 = Dpole4(output2); output2a = Dpole5(output1) ;     output6 = Dpole3(output2a) ;      output7 = Dpole4(output2a + output6)
            
            ###------ FAR RIGHT MOVE ::: OUTPUT NODE::    ##---- DOWN MOVE ---#####               ###---- DOWN THEN LEFT TURN AND UP----###
            output10 = Dpole5(output8 + output7)   ;    output9 = Dpole3(output10) ;   output11 = Dpole4(output9)
            
            ##---- UPWARD MOVE COVERGE TWO NODES---####            ###---- MOVING UP AND MULTIPLE WAYS --#######
            output4 = Dpole5(output9 + output11 ) ;   output2a = Dpole3(output4 + output2 )
            
            ###----- RIGHT DOWN RIGHT MOVE ---######                 
            output5a = Dpole4(output2a + output5)
            
            ##--- UP MOVE ----####                                   ###-----RIGHT DOWN RIGHT ---########
            output3a = Dpole5(output5a + output3) ; output8a = Dpole3(output8 + output3a)
            
             ###------ FROM NODE 8 - 10 DOWN MOVE ---###########
            output10a = Dpole4(output8a + output10)
           
            return output10a          
            




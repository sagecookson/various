# Perceptron Learning Algorithm as given in notes 1a).

import pandas as pd

# Training Examples (Dictionary of ordered pairs (x1,x2))
tr = {'a':(0,1),'b':(2,0),'c':(1,1)}

# Class
cl = {'a':-1, 'b':-1, 'c':1}

# Learning rate
lr = 1

# Initial weights
w0 = -1.5
w1 = 0
w2 = 2

# Initialise lists which will make up Dataframe after training
w0l=[]
w1l=[]
w2l=[] 
item=[]
x1=[] 
x2=[]
cla = []
s = []
fine = 0 # Used to count number of iterations not requiring any weight adjustment

# Train perceptron. While loop terminates when all training examples can be 
# iterated through without adjustment to weight. 
while fine < len(tr):
    fine = 0 
    for key in tr:
        w0l.append(w0)
        w1l.append(w1)
        w2l.append(w2)
        item.append(key)
        x1.append(tr[key][0])
        x2.append(tr[key][1])
        cla.append(cl[key]/abs(cl[key]))
        r = w0 + w1*tr[key][0] + w2*tr[key][1]
        if r >0 and cl[key] < 0:
            w0 = w0 - lr
            w1 = w1 - lr*tr[key][0]
            w2 = w2 - lr*tr[key][1]
        elif r < 0 and cl[key] > 0:
            w0 = w0 + lr
            w1 = w1 + lr*tr[key][0]
            w2 = w2 + lr*tr[key][1]
        else:
            fine = fine + 1

        s.append(r)

# Create DataFrame which resembles 1a) Q2 result, and print.
data = pd.DataFrame({'w0': w0l, 'w1':w1l, 'w2':w2l, 'item':item, 'x1':x1, 'x2':x2, 'class':cla, 's':s})
print(data)
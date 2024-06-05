import numpy as np
import pandas as pd
import math

def gredient_descendent(x,y):
    m_curr = b_curr = 0
    n= len(x)
    iteration = 500000
    learning_rate = 0.0002
    cost_prev= 0

    for i in range(iteration):
        y_predicted = (m_curr* x)+ b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)
        
        m_curr = m_curr - (learning_rate * md)
        b_curr = b_curr - (learning_rate * bd)

        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

        if(math.isclose(cost_prev,cost,rel_tol=1e-20,abs_tol=0.0)):
            print("done")
            break
        cost_prev = cost


df = pd.read_csv("test_scores.csv")
x= df.math
y= df.cs

gredient_descendent(x,y)

import numpy as np
import math

def transform(M,i):
    # dim=M.shape
    rowsinM=len(M)
    xref=M[i][0]
    yref=M[i][1]
    thetaref=M[i][3]
    t=np.zeros((rowsinM,4), dtype=int)
    r=np.array([[math.cos(thetaref),math.sin(thetaref),0],[-(math.sin(thetaref)),math.cos(thetaref),0],[0,0,1]])
    # print r
    for i in range(0, rowsinM):
        b=np.array([M[i][0]-xref,M[i][1]-yref,M[i][3]-thetaref])
        t[i, 0:3]=b.dot(r)
        t[i,3:4]=M[i][2]
    return t

# minu=np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
# print minu,"\n"
# y=transform(minu,2)
# print y

# t=np.zeros((4,4), dtype=int)
# print t,"\n"
# x=t[0:4,3:4]
# print x
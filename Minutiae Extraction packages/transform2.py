import numpy as np
import math
def transform2(T,alpha):
    # dim=T.shape
    rowsinT=len(T)
    tnew=np.zeros((rowsinT,4), dtype=int)
    r = np.array([[math.cos(alpha), math.sin(alpha), 0,0], [-(math.sin(alpha)), math.cos(alpha), 0,0], [0, 0, 1,0],[0,0,0,1]])
    for i in range(0,rowsinT):
        b=np.array([T[i,0:4]-[0,0,alpha,0]])
        tnew[i,0:4]=b.dot(r)
    return tnew

# minu=np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
# # print minu,"\n"
# # print minu[0,0:4]
# # y=transform2(minu,2)
# # print y
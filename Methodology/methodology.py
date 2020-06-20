from PIL import Image, ImageDraw
import utils
import argparse
import math
import os
import cv2
import skimage
import numpy as np
import skimage.morphology
from getTerminationBifurcation import getTerminationBifurcation
from removeSpuriousMinutiae import removeSpuriousMinutiae
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from random import sample
import pywt
"""
LOC for the functions used to find the core delta and whorl points 
"""

signum = lambda x: -1 if x < 0 else 1

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def get_angle(left, right):
    angle = left - right
    if abs(angle) > 180:
        angle = -1 * signum(angle) * (360 - abs(angle))
    return angle

def poincare_index_at(i, j, angles, tolerance):
    deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
    index = 0
    for k in range(0, 8):
        if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
            deg_angles[k + 1] += 180
        index += get_angle(deg_angles[k], deg_angles[k + 1])

    if 180 - tolerance <= index and index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index and index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index and index <= 360 + tolerance:
        return "whorl"
    return "none"

def calculate_singularities(im, angles, tolerance, W):
    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)
    colors = {"loop" : (150, 0, 0), "delta" : (0, 150, 0), "whorl": (0, 0, 150)}
    c=[]
    d=[]
    w=[]
    for i in range(1, len(angles) - 1):
        for j in range(1, len(angles[i]) - 1):
            singularity = poincare_index_at(i, j, angles, tolerance)
            if singularity != "none":
                draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)], outline = colors[singularity])
                # print i,j
                if (i>4 and i<15) and (j>3 and j<19) and  singularity=='loop' :
                    c.append([i,j])
                if (i>4 and i<15) and (j>3 and j<19) and  singularity=='delta' :
                    d.append([i,j])
                if (i>4 and i<15) and (j>3 and j<19) and  singularity=='whorl' :
                    w.append([i,j])
    del draw
    for i in range(0,len(c),4):
        x=int((c[i][0]+0.5)*16)
        y=int((c[i][1]+0.5)*16)
        angle = np.rad2deg(np.arctan2(c[i][0] - 0, c[i][1] - 0))
        core.append([x,y,angle])
    for i in range(0,len(d),4):
        x=int((d[i][0]+0.5)*16)
        y=int((d[i][1]+0.5)*16)
        angle = np.rad2deg(np.arctan2(d[i][0] - 0, d[i][1] - 0))
        delta.append([x,y,angle])
    for i in range(0,len(w),4):
        x=int((w[i][0]+0.5)*16)
        y=int((w[i][1]+0.5)*16)
        angle = np.rad2deg(np.arctan2(w[i][0] - 0, w[i][1] - 0))
        whorl.append([x,y,angle])

    return result



"""
LOC to find the core delta and whorl points using poincare index implemented in the above functions and used below
"""

parser = argparse.ArgumentParser(description="Singularities with Poincare index")
parser.add_argument("image", nargs=1, help = "Path to image")
parser.add_argument("block_size", nargs=1, help = "Block size")
parser.add_argument("tolerance", nargs=1, help = "Tolerance for Poincare index")
parser.add_argument('--smooth', "-s", action='store_true', help = "Use Gauss for smoothing")
parser.add_argument("--save", action='store_true', help = "Save result image as src_poincare.gif")
args = parser.parse_args()

im = Image.open(args.image[0])
im = im.convert("L")  # convert to grayscale
# im.show()
core = []
delta=[]
whorl=[]

W = int(args.block_size[0])

f = lambda x, y: 2 * x * y
g = lambda x, y: x ** 2 - y ** 2

angles = utils.calculate_angles(im, W, f, g)
if args.smooth:
    angles = utils.smooth_angles(angles)

result = calculate_singularities(im, angles, int(args.tolerance[0]), W)
# result.show()
img_name = 'poincare.tif'
#cv2.imwrite(r'C:\Users\pratimasharma\PycharmProjects\fingerprintCODES\enhanced\enh' + img_name,(result))
# result.save(r'C:\Users\pratimasharma\PycharmProjects\fingerprintCODES\enhanced\2.tif')
# print(core)
# print(delta)
# print(whorl)

extrapoints=5
if len(core)>0:
    extrapoints=extrapoints-len(core)
if len(delta)>0:
    extrapoints=extrapoints-len(delta)
if len(whorl)>0:
    extrapoints=extrapoints-len(whorl)

"""
LOC to find the minutia points
"""

if args.save:
    base_image_name = os.path.splitext(os.path.basename(args.image[0]))[0]
    result.save(base_image_name+ "_poincare.tif", "TIF")

img=cv2.imread(args.image[0],0)
img = np.uint8(img > 128);
# print img.shape
skel = skimage.morphology.skeletonize(img)
skel = np.uint8(skel) * 255;

mask = img * 255;
(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);

minutiaeTerm = skimage.measure.label(minutiaeTerm, 8);
RP = skimage.measure.regionprops(minutiaeTerm)
minutiaeTerm = removeSpuriousMinutiae(RP, np.uint8(img), 10);

BifLabel = skimage.measure.label(minutiaeBif, 8);
TermLabel = skimage.measure.label(minutiaeTerm, 8);

minutiaeBif = minutiaeBif * 0;
minutiaeTerm = minutiaeTerm * 0;

# Skeletonization or thinning of probe image

(rows, cols) = skel.shape
DispImg = np.zeros((rows, cols, 3), np.uint8)
DispImg[:, :, 0] = skel;
DispImg[:, :, 1] = skel;
DispImg[:, :, 2] = skel;
# cv2.imshow('skel', DispImg)

“””
Extracting minutia coordinates and orientation
“””
combset = []
RP = skimage.measure.regionprops(BifLabel)
for i in RP:
    (row, col) = np.int16(np.round(i['Centroid']))
    minutiaeBif[row, col] = 1;
    (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
    skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0));
    combset.append((row, col))

RP = skimage.measure.regionprops(TermLabel)
for i in RP:
    (row, col) = np.int16(np.round(i['Centroid']))
    minutiaeTerm[row, col] = 1;
    (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
    skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255));
    combset.append((row, col))
# print combset

for i in core :
    if len(core)>0:
        combset.append([i[0],i[1]])

for i in delta:
    if len(delta)>0:
        combset.append([i[0],i[1]])

for i in whorl:
    if len(core)>0:
        combset.append([i[0],i[1]])
print combset

"""
LOC to store original features from image
"""
# save = ""
# for i in combset:
#     for j in i:
#         save = save + str(j) + "\n"
# text_file = open(r"F:\PrathamProject\fingerprintproject\enhanceddataset\user1\enh1b.txt","a")
# n = text_file.write(save)
# text_file.close()


"""
LOC to find the delaunay graph
"""
points = np.array(combset)
tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
# plt.show()

"""
LOC to find the coordinates of delaunay triangles
"""
coordinatesoftriangles=[]
for i in tri.simplices:
    coordinatesoftriangles.append([[points[i][0][0],points[i][0][1]],[points[i][1][0],points[i][1][1]],[points[i][2][0],points[i][2][1]]])
# print coordinatesoftriangles

"""
LOC to find the coordinates near core points in the delaunay graph
"""
coregraph=[]
for i  in coordinatesoftriangles:
    for j in i:
        for k in core:
            if j[0]==k[0] and j[1]==k[1]:
                coregraph.append(i)
# print coregraph

coresurround=[]
for i in coregraph:
    for j in i:
        for k in core:
            if j[0]!=k[0] or j[1]!=k[1]:
                if len(coresurround)>0:
                    flag=0
                    for l in coresurround:
                        if j[0]==l[0] and j[1]==l[1]:
                            flag=1
                    if flag==0:
                        angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                        coresurround.append([j[0], j[1],angle])
                else:
                    angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                    coresurround.append([j[0],j[1],angle])
# print coresurround


"""
LOC to find the coordinates near delta points in the delaunay graph
"""

deltagraph=[]
for i  in coordinatesoftriangles:
    for j in i:
        for k in delta:
            if j[0]==k[0] and j[1]==k[1]:
                deltagraph.append(i)
# print coregraph

deltasurround=[]
for i  in deltagraph:
    for j in i:
        for k in delta:
            if j[0]!=k[0] and j[1]!=k[1]:
                if len(deltasurround)>0:
                    flag=0
                    for l in deltasurround:
                        if j[0]==l[0] and j[1]==l[1]:
                            flag=1
                    if flag==0:
                        angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                        deltasurround.append([j[0],j[1],angle])
                else:
                    angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                    deltasurround.append([j[0],j[1],angle])


# print deltasurround


"""
LOC to find the coordinates near whorl points in the delaunay graph
"""

whorlgraph=[]
for i  in coordinatesoftriangles:
    for j in i:
        for k in whorl:
            if j[0]==k[0] and j[1]==k[1]:
                whorlgraph.append(i)
# print coregraph

whorlsurround=[]
for i  in whorlgraph:
    for j in i:
        for k in whorl:
            if j[0]!=k[0] and j[1]!=k[1]:
                if len(whorlsurround)>0:
                    flag=0
                    for l in whorlsurround:
                        if j[0]==l[0] and j[1]==l[1]:
                            flag=1
                    if flag==0:
                        angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                        whorlsurround.append([j[0],j[1],angle])
                else:
                    angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                    whorlsurround.append([j[0],j[1],angle])


# print whorlsurround

"""
LOC to find vci index using density of node distance between pair of cords and diff of orientation angle
"""
def calcdelta(a, b):
    if a > b:
        ang = (a - b)
    else:
        ang = (360 + a) - b
    return (ang)


def dist(a, b, c, d):
    return math.sqrt((a - c) ** 2 + (b - d) ** 2)

def calcdensity(x):
    xgraph = []
    for i in coordinatesoftriangles:
        for j in i:
            if j[0] == x[0] and j[1] == x[1]:
                xgraph.append(i)
    # print coregraph

    xsurround = []
    for i in xgraph:
        for j in i:
            if j[0] != x[0] and j[1] != x[1]:
                if len(xsurround) > 0:
                    flag = 0
                    for l in xsurround:
                        if j[0] == l[0] and j[1] == l[1]:
                            flag = 1
                    if flag == 0:
                        angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                        xsurround.append([j[0], j[1], angle])
                else:
                    angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                    xsurround.append([j[0], j[1], angle])
    return len(xsurround)

def calcsurround(x):
    xgraph = []
    for i in coordinatesoftriangles:
        for j in i:
            if j[0] == x[0] and j[1] == x[1]:
                xgraph.append(i)
    # print coregraph

    xsurround = []
    for i in xgraph:
        for j in i:
            if j[0] != x[0] and j[1] != x[1]:
                if len(xsurround) > 0:
                    flag = 0
                    for l in xsurround:
                        if j[0] == l[0] and j[1] == l[1]:
                            flag = 1
                    if flag == 0:
                        angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                        xsurround.append([j[0], j[1], angle])
                else:
                    angle = np.rad2deg(np.arctan2(j[0] - 0, j[1] - 0))
                    xsurround.append([j[0], j[1], angle])
    return xsurround

"""
LOC to find the most dense point in graph
"""
denarr=[]
dencord=[]
for i in coordinatesoftriangles:
    for j in i:
        flagc = 0
        flagd = 0
        flagw = 0
        for l in core:
            if j[0]==l[0] and j[1]==l[1]:
                flagc=1
        for l in delta:
            if j[0]==l[0] and j[1]==l[1]:
                flagd=1
        for l in whorl:
            if j[0]==l[0] and j[1]==l[1]:
                flagw=1
        if flagc==0 and flagd==0 and flagw==0:
            x=calcdensity(j)
            if len(denarr)>0:
                flag=0
                for k in denarr:
                    if x==k:
                        flag=1
                if flag==0:
                    denarr.append(x)
                    dencord.append(j)
            else:
                denarr.append(x)
                dencord.append(j)

print denarr,dencord

dict ={}

for i in range(len(denarr)):
    for j in range(len(dencord)):
        if i==j:
            dict[denarr[i]]=dencord[j]

k = dict.keys()
length=len(k)-1
features=[]
if len(dict)>=extrapoints:
    while extrapoints!=0:
        for i in range(length , 0, -1):
            features.append(dict.get(k[i]))
            break
        length=length-1
        extrapoints=extrapoints-1
else:
    for i in range(length, 0, -1):
        features.append(dict.get(k[i]))
print features

"""
LOC to find the vci feature vector
"""

minden = 200
mind = 3600
mintheta = 360
maxden = 0
maxd = 0
maxtheta = 0
vci = []

for i in coresurround:
    for j in core:
        den=calcdensity(i)
        d=dist(j[0],j[1],i[0],i[1])
        theta=calcdelta(j[2],i[2])
        if(minden>den):
            minden=den
        if (mind > d):
            mind = d
        if (mintheta > theta):
            mintheta = theta
        if (maxden < den):
            maxden = den
        if (maxd < d):
            maxd = d
        if (maxtheta < theta):
            maxtheta = theta

        vci.append([int(math.ceil(den)), int(math.ceil(d)), int(math.ceil(theta))])


for i in deltasurround:
    for j in delta:
        den=calcdensity(i)
        d=dist(j[0],j[1],i[0],i[1])
        theta=calcdelta(j[2],i[2])
        if(minden>den):
            minden=den
        if (mind > d):
            mind = d
        if (mintheta > theta):
            mintheta = theta
        if (maxden < den):
            maxden = den
        if (maxd < d):
            maxd = d
        if (maxtheta < theta):
            maxtheta = theta


        vci.append([int(math.ceil(den)), int(math.ceil(d)), int(math.ceil(theta))])
       

for i in whorlsurround:
    for j in whorl:
        den=calcdensity(i)
        d=dist(j[0],j[1],i[0],i[1])
        theta=calcdelta(j[2],i[2])
        if(minden>den):
            minden=den
        if (mind > d):
            mind = d
        if (mintheta > theta):
            mintheta = theta
        if (maxden < den):
            maxden = den
        if (maxd < d):
            maxd = d
        if (maxtheta < theta):
            maxtheta = theta

        
        vci.append([int(math.ceil(den)), int(math.ceil(d)), int(math.ceil(theta))])
        

for i in features:
    surround=calcsurround(i)
    print surround,len(surround)
    for j in surround:
        den = calcdensity(j)
        d = dist(j[0], j[1], i[0], i[1])
        angle = np.rad2deg(np.arctan2(i[0] - 0, i[1] - 0))
        theta = calcdelta(j[2], angle)
        if (minden > den):
            minden = den
        if (mind > d):
            mind = d
        if (mintheta > theta):
            mintheta = theta
        if (maxden < den):
            maxden = den
        if (maxd < d):
            maxd = d
        if (maxtheta < theta):
            maxtheta = theta


        vci.append([int(math.ceil(den)), int(math.ceil(d)), int(math.ceil(theta))])
        

print vci,len(vci)

"""
LOC for modulo exponentiation
"""

modulo = []

for i in vci:
    temp = []
    for j in i:
        temp.append(pow(j, 2, 397))
    #     primitive root of 397 is 5 do (5^x)^2 mod 397
    modulo.append(temp)
vci = modulo


"""
LOC for quantisation 
"""

stepden = (math.ceil(abs((maxden-minden)) ))
stepd = (math.ceil(abs((maxd-mind)/3) ))
steptheta = (math.ceil(abs((maxtheta-mintheta)/3)))

print stepden,stepd,steptheta
sden = math.ceil(20 / stepden)
sd = math.ceil(300 / stepd)
stheta = math.ceil(360 / steptheta)

print sden,sd,stheta

bsl = int((sden) * (sd) * (stheta))



# print math.ceil(sl),math.ceil(sphi),math.ceil(sd)
vcindex = []

for i in vci:
        a = i[0] / stepden
        b =  i[1]/ stepd
        c =  i[2]/ steptheta
        vcindex.append((int(math.floor(a)), int(math.floor(b)), int(math.floor(c))))
vcindex.sort()
print vcindex


"""
LOC for 
"""

indexl = []
maxval=0
k = 0
for i in range(len(vcindex)):
    ind = vcindex[k][2] * (sd) + (vcindex[k][0] * (sden) + vcindex[k][1])  # z.Y + (x.X+y)
    indexl.append(int(ind))
    if maxval<int(ind):
        maxval=int(ind)
    k = k + 1
indexl.sort()
print indexl

blen = maxval+1
print blen

bitstring = np.zeros(blen, dtype=int)

# print indexl
for i in indexl:
    bitstring[i] = 1

bitt = np.array([bitstring])
# print bitt, bitt.shape
# print dbv



"""
LOC for partial DFT
"""


def dftmtx(N):
    return scipy.fftpack.fft2(scipy.eye(N))
W = dftmtx(blen)
ctemplate = W * bitt
print ctemplate


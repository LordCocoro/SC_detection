import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.image as mpimg
import os,cv2
import xml.etree.ElementTree as ET
tree = ET.parse('Dataset/annotation/004.xml')
img = mpimg.imread('Dataset/rename/004.png')
print(img)
root = tree.getroot()
print(root[2].attrib)
fig, ax = plt.subplots()

ax.imshow(img)
x_loss = 0.0
y_loss = 0.0
for p in root[2]:
    setPx = lambda px: px*3.7795275591
    if(p.attrib['id'].startswith('imag')):
        x_loss = float(p.attrib['x'])
        y_loss = float(p.attrib['y'])
        print(x_loss,y_loss)
    print(x_loss,y_loss)
    getx = lambda x: (x-x_loss) * 3.7795275591/32
    gety = lambda y: (y-y_loss) * 3.7795275591/24
    if(p.attrib['id'].startswith('rec')):
        _x=getx(float(p.attrib['x']))
        _y=gety(float(p.attrib['y']))
        _w=setPx(float(p.attrib['width']))/32
        _h=setPx(float(p.attrib['height']))/24
        x_min = _x
        y_min = _y
        x_max = _x + _w
        y_max = _y + _h
        print(x_min,x_max,y_min,y_max)
        if(p.attrib['class'].startswith('proto')):
            rect = patches.Rectangle((_x,_y),_w,_h, linewidth=1, edgecolor='r', facecolor='none')
        if(p.attrib['class'].startswith('zerg')):
            rect = patches.Rectangle((_x,_y),_w,_h, linewidth=1, edgecolor='r', facecolor='none')
        if(p.attrib['class'].startswith('terran')):
            rect = patches.Rectangle((_x,_y),_w,_h, linewidth=1, edgecolor='r', facecolor='none')
        if(p.attrib['class'].startswith('gas')):
            rect = patches.Rectangle((_x,_y),_w,_h, linewidth=1, edgecolor='g', facecolor='none')
        if(p.attrib['class'].startswith('min')):
            rect = patches.Rectangle((_x,_y),_w,_h, linewidth=1, edgecolor='b', facecolor='none')

        ax.add_patch(rect)


# x1 = 128.05721
# y1 = 213.65707
# x2 = 148.05721
# y2 = 193.65707

# cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2)

# plt.figure()
plt.show()
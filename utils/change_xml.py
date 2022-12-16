import os
import shutil 
from os import listdir
from os.path import isfile, join
p = os.path.join('Dataset/', 'annotation')
    
if not os.path.exists(p):
    os.makedirs(p)
onlyfiles = [f for f in listdir('Dataset/sgv') if isfile(join('Dataset/sgv', f))]
onlyfiles = sorted(onlyfiles)
for e in range(len(onlyfiles)):
    #print(onlyfiles[e])
    desde = 'Dataset/sgv/'+onlyfiles[e]
    print(desde,onlyfiles[e].split('.')[0] )
    hasta = 'Dataset/annotation/'+onlyfiles[e].split('.')[0]+'.xml'

    shutil.copyfile(desde,hasta)
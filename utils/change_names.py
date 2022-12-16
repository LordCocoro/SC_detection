import os
import shutil 
from os import listdir
from os.path import isfile, join
p = os.path.join('Dataset/', 'rename')
    
if not os.path.exists(p):
    os.makedirs(p)
onlyfiles = [f for f in listdir('Dataset/images_4_v2') if isfile(join('Dataset/images_4_v2', f))]

onlyfiles = sorted(onlyfiles)
for e in range(len(onlyfiles)):
    print(onlyfiles[e])
    desde = 'Dataset/images_4_v2/'+onlyfiles[e]
    if(e+79>=99):
        hasta = 'Dataset/rename/'+str(e+80)+'.png'
    else:
        hasta = 'Dataset/rename/0'+str(e+80)+'.png'
    shutil.copyfile(desde,hasta)
# for k in imdata:
#     desde = realdir+'/images/'+imdata[k].name
#     hasta = realdir+'/rename/'+imdata[k].name
#     shutil.copyfile(desde,hasta)
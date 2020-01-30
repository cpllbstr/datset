import matplotlib.pyplot as plt
import matplotlib.image as mimg
# import argparse
import sys

name = sys.argv[1]
file = open(name[:-4]+".txt")
line = file.readline()
parsed = line.split(r' ')

clss, x, y, w, h = parsed 

print(clss, x, y, w, h)

fig,ax = plt.subplots(1)
img = mimg.imread(name)
print(len(img), len(img[0]))
ax.imshow(img)
rect = plt.Rectangle(((float(x)-float(w)/2)*len(img), (float(y)-float(h)/2)*len(img[0])),float(w)*len(img), float(h)*len(img), linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()
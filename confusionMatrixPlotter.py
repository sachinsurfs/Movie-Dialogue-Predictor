import matplotlib.pyplot as plt
import numpy as np

conf_arr = [[1525,688],[  10,36]]
#conf_arr = [[2213,0],[  42,4]]
#conf_arr = [[10600,5765],[77,270]]


norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, 
                interpolation='nearest')

width, height = 2,2

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center',  fontsize=20)

cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), [0,1])
plt.yticks(range(height), [0,1])
plt.title('Confusion matrix')


'''
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
'''
plt.show()
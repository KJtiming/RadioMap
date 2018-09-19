import numpy as np
import math

set_feature = np.array([19,19,19,19,19,19], np.float)



# read row data
data = np.genfromtxt('set3_mod.csv', delimiter=',')

def convert_location_data(x, y) :
    '''
    lng = a * x1 - b * y2 + c 
    lat = a * x2 + b * y1 + d

    NEMO -> indoor position value
    (840, -351) -> (96.824, 0)
    (923, -179) -> (107.068, 15.874)
    '''
    a = 0.1185
    b = -0.003
    c = -1.81
    d = 39.842
 
    lng = a * x - b * y * (-1) + c - 4
    lat = a * y * (-1) + b * x + d - 0.5

    return lng, lat

# coordinate transform
a = 0.1185
b = -0.003
c = -1.81
d = 39.842

lng = a * data[:,0] - b * data[:,1] * (-1) + c - 4
lat = a * data[:,1] * (-1) + b * data[:,0] + d - 1
data[:,0] = lng
data[:,1] = lat

x1, y1 = convert_location_data (868, 199)
x2, y2 = convert_location_data (735, 206)
dis1_new = np.sqrt(np.power(lng-x1, 2)+np.power(lat-y1, 2))
dis2_new = np.sqrt(np.power(lng-x2, 2)+np.power(lat-y2, 2))
#data[:,14] = dis1_new
#data[:,15] = dis2_new
#print "data[:,15]==",data[:,15]
np.savetxt('data.csv', data, delimiter=',', fmt='%f')
# 
n_rem = np.zeros([106, 26])
for i in range(len(data)): 
    this_x = np.floor(data[i,0])
    this_y = np.floor(data[i,1])
    n_rem[int(this_x),int(this_y)] = 1
print "n_rem==",len(np.where(n_rem==1)[0])


# extract set data
idx_set = np.where((data[:,2:8] == set_feature).all(axis=1))[0]
#idx_set = np.where((data[:,2:-6] == set_feature).all(axis=1))[0]
#print "data[:,2:-2] ==",data[:,2:-6] 
#print "idx_set==",idx_set
set_data = data[idx_set,:]
#print "set_data==",set_data



# shuffle data
num_setdata = len(set_data)
idx_rnd = np.random.permutation(num_setdata)
shuf_data = set_data[idx_rnd, :]



# get the testing data
p_test = 0.1
n_test = round(num_setdata*p_test, 0)
d_test = shuf_data[0:int(n_test),:]
np.savetxt('rem_test.csv', d_test, delimiter=',', fmt='%f')



# get the pci training data
pci_feature = np.unique(data[:,0:-2], axis=0)
get_pci = np.zeros([len(pci_feature), 1])
for i in range(len(pci_feature)): 
    idx_data = np.where((data[:,0:-2] == pci_feature[i,:]).all(axis=1))[0] 
    counts = np.bincount(data[idx_data,-2].astype(int))
    get_pci[i] = np.argmax(counts)

pci_training = np.hstack((pci_feature, get_pci))
np.savetxt('rem_pci_train.csv', pci_training, delimiter=',', fmt='%f')



# get the rsrp training data
rsrp_feature = np.unique(data[:,0:-1], axis=0)
get_rsrp = np.zeros([len(rsrp_feature), 3])
for i in range(len(rsrp_feature)): 
    idx_data = np.where((data[:,0:-1] == rsrp_feature[i,:]).all(axis=1))[0] 
    get_rsrp[i,0] = np.min(data[idx_data,-1])
    get_rsrp[i,1] = np.max(data[idx_data,-1])
    get_rsrp[i,2] = np.mean(data[idx_data,-1])

rsrp_training = np.hstack((rsrp_feature, get_rsrp))
np.savetxt('rem_rsrp_train.csv', rsrp_training, delimiter=',', fmt='%f')



# print msg
print('num of testing data: '+str(len(d_test)))
print('num of training data for pci: '+str(len(pci_training)))
print('num of training data for rsrp: '+str(len(rsrp_training)))

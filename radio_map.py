from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from array import array
from keras.utils import to_categorical
from keras.utils import np_utils
from ann_visualizer.visualize import ann_viz


# read row data
pci_train = np.genfromtxt('rem_pci_train.csv', delimiter=',')
rsrp_train = np.genfromtxt('rem_rsrp_train.csv', delimiter=',')
data_test = np.genfromtxt('rem_test.csv', delimiter=',')



# select classifier and regressor
modelClassifier = KNeighborsClassifier(n_neighbors=3)
#modelClassifier = DecisionTreeClassifier()
#modelClassifier = MLPClassifier(hidden_layer_sizes=(300,), random_state=1, max_iter=1, warm_start=True)

modelRegressor = KNeighborsRegressor(n_neighbors=3)
#modelRegressor = DecisionTreeRegressor()
# modelRegressor = RandomForestRegressor()
# modelRegressor = GradientBoostingRegressor()
#modelRegressor = MLPRegressor()
# modelRegressor = SVR()


# create model
"""
model = Sequential()
model.add(Dense(12, input_dim=18,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='relu'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, shuffle=True)
#model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=16, shuffle=True)

# new instance where we do not know the answer
Xnew = data_test[:,0:-2]
predictions = model.predict(X)
np.savetxt('ynew_pci.csv', predictions, delimiter=',', fmt='%f')

# show the inputs and predicted outputs
#print("X=%s, Predicted=%s" % (Xnew[0], ynew_pci[0]))
"""



# pci prediction
pci_pred = modelClassifier.fit(pci_train[:,0:-1], pci_train[:,-1]).predict(data_test[:,0:-2])
#np.savetxt('pci_pred.csv', pci_pred, delimiter=',', fmt='%f')
n_accuracy = 0
for i in range(len(pci_pred)):
    if pci_pred[i] == data_test[i,-2]: 
        n_accuracy = n_accuracy+1
print "n_accuracy_pci==",n_accuracy
print "len(pci_pred)==",len(pci_pred)
print('pci accuracy: ' + str(n_accuracy//len(pci_pred)))



# rsrp range prediction
rsrp_pred = MultiOutputRegressor(modelRegressor).fit(rsrp_train[:,0:-3], rsrp_train[:,-3:-1]).predict(data_test[:,0:-1])
np.savetxt('rsrp_pred.csv', rsrp_pred, delimiter=',', fmt='%f')
#qq=np.zeros([len(rsrp_pred), 2])
n_accuracy = 0
for i in range(len(rsrp_pred)):
    #qq[i] = [rsrp_pred[i,1]-rsrp_pred[i,0]]
    if rsrp_pred[i,0] <= data_test[i,-1] and data_test[i,-1] <= rsrp_pred[i,1]: 
        n_accuracy = n_accuracy+1
print "n_accuracy==",n_accuracy
print "len(rsrp_pred)==",len(rsrp_pred)
print('rsrp accuracy: ' + str(n_accuracy/len(rsrp_pred)))


print "np.mean(qq)==",np.mean(rsrp_pred[:,1]-rsrp_pred[:,0])
print "np.mean(min)==",np.mean(rsrp_pred[:,0])
print "np.mean(max)==",np.mean(rsrp_pred[:,1])




# rsrp prediction
model_rsrp = modelRegressor
model_rsrp.fit(rsrp_train[:,0:-3], rsrp_train[:,-1]) 
rsrp_val = model_rsrp.predict(data_test[:,0:-1]) 
RMSE = np.sqrt(np.square(np.subtract(data_test[:,-1], rsrp_val)).mean())
print('RSRP RMSE: ' + str(RMSE))

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
#calculate distance and angle between location points and small cell
def cal_distance_to_cell (x_cell, y_cell, x_loc, y_loc) :
    dist = math.sqrt( (x_loc - x_cell)**2 + (y_loc - y_cell)**2 )
    #if x1>80:
    #    print "dist==",dist
    return dist

def cal_angle_to_cell (diff_x, diff_y) :
    radian = math.atan2(diff_y, diff_x)
    #print "radian==",radian
    degree = math.degrees(radian)
    #print "degree==",degree
    return degree

# radio map prediction
idx = 0
print "data_test.shape==",data_test.shape
pci_map_test = np.zeros([26*106, data_test.shape[1]-2])
#pci_map_test = np.zeros([26*106, data_test.shape[1]-6])
print "146pci_map_test==",pci_map_test
#map_feature = data_test[1,2:-6]
map_feature = data_test[1,2:-14]
#print "data_test[1,2:-6]==",data_test[1,2:-6]
#x1, y1 = convert_location_data (868, 199)
#x2, y2 = convert_location_data (735, 206)
x = np.zeros([6,1])
y = np.zeros([6,1])
x[0], y[0] = convert_location_data (260, 215)
x[1], y[1] = convert_location_data (480, 158)
x[2], y[2] = convert_location_data (630, 210)
x[3], y[3] = convert_location_data (710, 275)
x[4], y[4] = convert_location_data (765, 145)
x[5], y[5] = convert_location_data (908, 130)
print "x[0],y[0]==",x[0], y[0]
print "x[1],y[1]==",x[1], y[1]
print "x[2],y[2]==",x[2], y[2]
print "x[3],y[3]==",x[3], y[3]
print "x[4],y[4]==",x[4], y[4]
print "x[5],y[5]==",x[5], y[5]

cell = ['37','38','39','40','41','42']
dis = np.zeros([6,1])
ang = np.zeros([6,1])
for a in range(0, 106):
    for b in range(0, 26):
        xy = np.array([a, b], np.float)
        #print "xy==",xy
        #print "xy.shape==",xy.shape
        #print "xy.ndim==",xy.ndim
        for i in range(5) : 
            dis[i] = cal_distance_to_cell (x[i], y[i], xy[0], xy[1])
            ang[i] = cal_angle_to_cell (x[i]-xy[0], y[i]-xy[1])
        #ang1 = cal_angle_to_cell (xy[0]-x1, xy[0]-y1)
        #ang2 = cal_angle_to_cell (xy[0]-x2, xy[0]-y2)
        #if a==80 and b==18:
          #print "ang1==",ang1
          #print "ang2==",ang2
          #print "dis1==",dis1
          #print "dis2==",dis2
        if 45<=ang[i]<90:
            ang[i]=1
        if 0<=ang[i]<45:
            ang[i]=2
        if -45<=ang[i]<0:
            ang[i]=3
        if -90<=ang[i]<-45:
            ang[i]=4
        if -135<=ang[i]<-90:
            ang[i]=5
        if -180<=ang[i]<-135:
            ang[i]=6
        if 135<=ang[i]<180:
            ang[i]=7
        if 90<=ang[i]<135:
            ang[i]=8
        dis_np = np.array(dis[i])
        ang_np = np.array(ang[i])
        #if a==80 and b==18:
          #print "ang1==",ang1
          #print "ang2==",ang2
        add_feature = np.array([dis[i],ang[i]])
        #print "add_feature==",add_feature
        np.savetxt('dis.csv', dis, delimiter=',', fmt='%f')
        np.savetxt('ang.csv', ang, delimiter=',', fmt='%f')
        np.savetxt('xy.csv', xy, delimiter=',', fmt='%f')
        #print "data_test.shape[1]==",data_test.shape[1]
        if data_test.shape[1] == 4:
            pci_map_test[idx,:] = xy
        else: 
            pci_map_test[idx,:] = np.hstack((xy))
            #pci_map_test[idx,:] = np.hstack((xy, map_feature, add_feature))
            #pci_map_test[idx,:] = np.hstack((xy,map_feature,dis[0],dis[1],dis[2],dis[3],dis[4],dis[5],ang[0],ang[1],ang[2],ang[3],ang[4],ang[5]))
        #print "pci_map_test[idx,:]==",pci_map_test[idx,:]
        idx = idx+1
np.savetxt('pci_map_test.csv', pci_map_test, delimiter=',', fmt='%f')
X = pci_train[:,0:-1]
Y = pci_train[:,-1]
#X = rsrp_train[:,0:-3]
#Y = rsrp_train[:,-3]
#Y = np_utils.to_categorical(pci_train[:,-1])
np.savetxt('rsrp_train_x.csv', X, delimiter=',', fmt='%f')
np.savetxt('rsrp_train_y.csv', Y, delimiter=',', fmt='%f')
y_train = np.genfromtxt('rsrp_train_y.csv', delimiter=',')
a = np.array([])
for i in range (len(y_train)):
    y_train_data = y_train[i]
    if y_train_data == 120:
       y_train_data = 1
    elif y_train_data == 151:
       y_train_data = 2
    elif y_train_data == 154:
       y_train_data = 3
    elif y_train_data == 301:
       y_train_data = 4
    elif y_train_data == 302:
       y_train_data = 5
    else:
       y_train_data = 0
       #print "pci is not in range"
    a = np.append(a,[y_train_data])
    np.savetxt('y_train_data.csv', a, delimiter=',', fmt='%f')
y_train_input = np.genfromtxt('y_train_data.csv', delimiter=',')
np.savetxt('pci_train_Y.csv', y_train_input, delimiter=',', fmt='%f')
Y = np_utils.to_categorical(y_train_input)

def build_dNN_model(X_train, y_train):
    print "+++Deep NN"
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim = 6))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, init='uniform', activation='sigmoid'))
    #model.add(Dense(1))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.1,verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('losee')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    model.summary()
    return model
def pltLearningCurve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    #plt.show()

#model = build_dNN_model(X,Y)
#pltLearningCurve(history)
#print "pci_map_test==",pci_map_test
#Xnew = np.array(data_test[:,0:-2])
Xnew = np.array(pci_map_test)
X_rsrp = data_test[:,0:-1]
#print "Xnew==",Xnew
#for DNN output = model.predict_classes(Xnew)
#output = model.predict(Xnew)
#for DNN np.savetxt('output_pci.csv', output, delimiter=',', fmt='%f')
#for DNN output_pred = np.genfromtxt('output_pci.csv', delimiter=',')
'''#for DNN
a = np.array([])
for i in range (len(output_pred)):
    output_y = output_pred[i]
    if output_y == 1:
       output_y = 120
    elif output_y == 2:
       output_y = 151
    elif output_y == 3:
       output_y = 154
    elif output_y == 4:
       output_y = 301
    elif output_y == 5:
       output_y = 302
    else:
       output_y = 1000 
    #print "a==",a
    a = np.append(a,[output_y])
    np.savetxt('output_y.csv', a, delimiter=',', fmt='%f')


'''
#ann_viz(model, title="model", view=True, filename="model.gv")
#print(pci_map_test)
#print "pci_train[:,0:-1]==",pci_train[:,0:-1]
pci_map_pred = modelClassifier.fit(pci_train[:,0:-1], pci_train[:,-1]).predict(pci_map_test)
np.savetxt('pci_map_pred.csv', pci_map_pred, delimiter=',', fmt='%f')
pci_map_pred_np = np.array([pci_map_pred])
a_np = np.array([a])

#print "pci_map_pred_np[:2]==",pci_map_pred_np[:2]
np.savetxt('qq.csv', pci_map_pred_np, delimiter=',', fmt='%f')

rsrp_map_test = np.hstack((pci_map_test, pci_map_pred_np.T))
#print "323rsrp_map_test==",rsrp_map_test
#rsrp_map_test = np.hstack((pci_map_test, a_np.T))

rsrp_map_pred = model_rsrp.predict(rsrp_map_test) 
rsrp_map_pred_np = np.array([rsrp_map_pred])
rsrp_map = np.hstack((pci_map_test, rsrp_map_pred_np.T))

#np.savetxt('pci_map.csv', rsrp_map_test, delimiter=',', fmt='%d')
#np.savetxt('rsrp_map.csv', rsrp_map, delimiter=',', fmt='%f')


# plot radio map
pci_plot = np.zeros([26, 106])
rsrp_plot = np.zeros([26, 106])

for i in range(len(rsrp_map_test)): 
    this_w = int(rsrp_map_test[i,0])
    this_h = int(rsrp_map_test[i,1])
    pci_z = rsrp_map_test[i,-1]
    
    if pci_z == 37: 
        pci_plot[this_h, this_w] = 10
    elif pci_z == 38: 
        pci_plot[this_h, this_w] = 35
    elif pci_z == 39: 
        pci_plot[this_h, this_w] = 75
    elif pci_z == 40: 
        pci_plot[this_h, this_w] = 20
    elif pci_z == 41: 
        pci_plot[this_h, this_w] = 65
    elif pci_z == 42: 
        pci_plot[this_h, this_w] = 95
    elif pci_z == 120: 
        pci_plot[this_h, this_w] = 85
    elif pci_z == 154: 
        pci_plot[this_h, this_w] = 85
    elif pci_z == 151: 
        pci_plot[this_h, this_w] = 85
    else: 
        pci_plot[this_h, this_w] = 100

    rsrp_z = rsrp_map[i,-1]
    rsrp_plot[this_h, this_w] = rsrp_z   

#KNN
def build_knn_model(X_train, y_train):
    print "+++KNN"
    model =  KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')
    #model =  KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto')
    model.fit(X_train, y_train)
    #print "y_train==",y_train
    return model
X = pci_train[:,0:-1]
Y = pci_train[:,-1]
#model = build_knn_model(X,Y)
map_size = [106, 26] 
x_resolution = map_size[0]
y_resolution = map_size[1]
pci = []
    #Get the maxium output
    #for i in range(len(output)) :
    #    print output[i]
    #raw_input()
'''
output = model.predict(pci_map_test)
np.savetxt('model.predict(pci_map_test).csv', output, delimiter=',', fmt='%f')

np.savetxt('pci.csv', pci, delimiter=',', fmt='%f')

z = np.reshape(output, (y_resolution, x_resolution))
z[:] = map(list,zip(*z[::-1]))

#z_pred = np.hstack((xy, pci_map_pred))
#z = np.reshape(z_pred, (y_resolution+1, x_resolution+1))
for j in range(y_resolution):
        for i in range(x_resolution) : 
            pci_z = z[j][i]
            if pci_z == 37 :#37
                plt.plot(round(i), round(j), color='blue', marker = 's', markersize=5, alpha=.1)
            elif pci_z == 38 :#38
                plt.plot(round(i), round(j), color='green', marker = 's', markersize=5, alpha=.1 )
            elif pci_z == 39 :#39
                plt.plot(round(i), round(j), color='red', marker = 's', markersize=5, alpha=.1 )
            elif pci_z == 40:#40
                plt.plot(round(i), round(j), color='chocolate', marker = 's', markersize=5, alpha=.1)
            elif pci_z == 41:#41
                plt.plot(round(i), round(j), color='skyblue', marker = 's', markersize=5, alpha=.1)
            elif pci_z == 42:#42
                plt.plot(round(i), round(j), color='yellow', marker = 's', markersize=5, alpha=.1)
            else :                 
                pci_z = 1000
                plt.plot(round(i), round(j), color='black', marker = 's', markersize=5, alpha=.1 )

img = plt.imread("./pic/51_5F-3.png")
plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
plt.savefig('test', dpi=200)
'''
fig_pci = plt.pcolor(pci_plot, vmin=1, vmax=100, cmap='gist_ncar')
#plt.colorbar(heatmap_pci)
#plt.axis([0, 35, 0, 25])
plt.axis('equal')
plt.axis('off')
plt.savefig('pci_map.png')
#plt.show()

fig_rsrp = plt.pcolor(rsrp_plot, vmin=-130, vmax=-80, cmap='jet')
plt.axis('equal')
plt.axis('off')
plt.savefig('rsrp_map.png')

im_pci = Image.open('pci_map.png')
im_rsrp = Image.open('rsrp_map.png')

bk = Image.open('5f_map.png')
blended = Image.blend(im_pci, bk, alpha=0.4)
blended.save('5f_pci_map.png')
blended.show()

blended = Image.blend(im_rsrp, bk, alpha=0.4)
blended.save('5f_rsrp_map.png')
blended.show()



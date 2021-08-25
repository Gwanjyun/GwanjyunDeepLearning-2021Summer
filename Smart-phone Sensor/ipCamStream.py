import json
import time
import numpy as np
import requests
import ctypes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from Quaternion import *
url = 'http://192.168.1.108:8080/sensors.json'
def getData(url = 'http://192.168.1.108:8080/sensors.json'):
    s = requests.Session()
    r = s.get(url)
    d = r.json()
    return d

class SensorData:
    def __init__(self, url):
        self.url = url
        self.data = ctypes.POINTER(ctypes.py_object)
        self.data.contents = {}
        
        self.t = threading.Thread(target = self.Listen, args = (self.data,))
        self.t.start()
        while True:
            if self.data.contents != {}:
                 break
           
    def getData(self,url):
        s = requests.Session()
        r = s.get(url)
        d = r.json()
        return d
    
    
    def Listen(self, data_pointer):
        while True:
            data_pointer.contents = self.getData(self.url)
d = SensorData(url)
t = d.data.contents['gravity']['data'][-1][0]


r0 = Quaternion(*d.data.contents['rot_vector']['data'][-1][1][:4]).inverse()
g = Quaternion(0,0,9.8)
o = Quaternion()
p1 = Quaternion(-5,-15,0)
p2 = Quaternion(-5,15,0)
p3 = Quaternion(5,15,0)
p4 = Quaternion(5,-15,0)

p5 = Quaternion(-5,-15,2)
p6 = Quaternion(-5,15,2)
p7 = Quaternion(5,15,2)
p8 = Quaternion(5,-15,2)

i = 0
x = []
g1 = []
g2 = []
g3 = []

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')

while True:
    # if d.data.contents['gravity']['data'][-1][0] - t > 0:
    if True:
        i+=1
        x.append(i)
        g1.append(d.data.contents['gravity']['data'][-1][1][0])
        g2.append(d.data.contents['gravity']['data'][-1][1][1])
        g3.append(d.data.contents['gravity']['data'][-1][1][2])
        # la1.append(myData.data['linearAcceleration']['value'][0])
        # la2.append(myData.data['linearAcceleration']['value'][1])
        # la3.append(myData.data['linearAcceleration']['value'][2])

        gx = g1[-1]
        gy = g2[-1]
        gz = g3[-1]
        g_v = Quaternion(gx, gy, gz)

        
        r = getRotateQuaternion(g,g_v)
        r = Quaternion(*d.data.contents['rot_vector']['data'][-1][1][:4])
        p1_ = p1.Rotate(r).Rotate(r0)
        p2_ = p2.Rotate(r).Rotate(r0)
        p3_ = p3.Rotate(r).Rotate(r0)
        p4_ = p4.Rotate(r).Rotate(r0)

        p5_ = p5.Rotate(r).Rotate(r0)
        p6_ = p6.Rotate(r).Rotate(r0)
        p7_ = p7.Rotate(r).Rotate(r0)
        p8_ = p8.Rotate(r).Rotate(r0)



        t = d.data.contents['gravity']['data'][-1][0]
        print(d.data.contents['gravity']['data'][-1][1])

        ax.clear()
        ax.plot([0,gx], [0,gy], [0,-gz], label='parametric curve')
        ax.plot([p1_.x, p2_.x, p3_.x, p4_.x, p1_.x],[p1_.y, p2_.y, p3_.y, p4_.y, p1_.y],[p1_.z, p2_.z, p3_.z, p4_.z, p1_.z],
        label='parametric curve')
        ax.plot([p5_.x, p6_.x, p7_.x, p8_.x, p5_.x],[p5_.y, p6_.y, p7_.y, p8_.y, p5_.y],[p5_.z, p6_.z, p7_.z, p8_.z, p5_.z],
        label='parametric curve')
        ax.plot([p1_.x,p5_.x],[p1_.y,p5_.y],[p1_.z,p5_.z])
        ax.plot([p2_.x,p6_.x],[p2_.y,p6_.y],[p2_.z,p6_.z])
        ax.plot([p3_.x,p7_.x],[p3_.y,p7_.y],[p3_.z,p7_.z])
        ax.plot([p4_.x,p8_.x],[p4_.y,p8_.y],[p4_.z,p8_.z])

        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_xlim3d(-10,10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-10,10)
        plt.pause(0.06)



        # ax.clear()
        # ax.plot(x[-50:],g1[-50:],'r-')
        # ax.plot(x[-50:],g2[-50:],'g-')
        # ax.plot(x[-50:],g3[-50:],'b-')

        # ax1.clear()
        # ax1.plot(x[-50:],la1[-50:],'r-')
        # ax1.plot(x[-50:],la2[-50:],'g-')
        # ax1.plot(x[-50:],la3[-50:],'b-')

        # plt.show()


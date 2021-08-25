import socket
import json
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Quaternion:
    def __init__(self, x = 0, y = 0, z = 0, s = 0):
        self.mean = (s**2 + x**2 + y**2 + z**2)**0.5
        self.s,self.x,self.y,self.z = s,x,y,z
        self.RotationMatrix()
        
    def Q2E(self, x, y, z, s = 0):
        self.__init__(x,y,z,s)
        return self
    
    def E2Q(self, phi, theta, Phi):
        self.phi,self.theta,self.Phi = phi, theta, Phi
        self.s = np.cos(phi/2)*np.cos(theta/2)*np.cos(Phi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(Phi/2)
        self.x = np.sin(phi/2)*np.cos(theta/2)*np.cos(Phi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(Phi/2)
        self.y = np.cos(phi/2)*np.sin(theta/2)*np.cos(Phi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(Phi/2)
        self.z = np.cos(phi/2)*np.cos(theta/2)*np.sin(Phi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(Phi/2)
        self.RotationMatrix()
        return self
    
    def RotationMatrix(self):
        s,x,y,z = self.s,self.x,self.y,self.z
        mean = (s**2 + x**2 + y**2 + z**2)**0.5
        if mean == 0:
            mean = 1
        s,x,y,z = s/mean,x/mean,y/mean,z/mean
        self.phi = np.arctan2(2*(s*x + y*z), 1 - 2*(x**2 + y**2))
        self.theta = np.arcsin(2*(s*y - z*x))
        self.Phi = np.arctan2(2*(s*z + x*y), 1 - 2*(y**2 + z**2))
        
        self.R = np.array([[1-2*(y**2 + z**2),     2 * (x*y - s*z),     2 * (s*y + x*z)],
                           [  2 * (x*y + s*z), 1 - 2*(x**2 + z**2),     2 * (y*z - s*x)],
                           [  2 * (x*z - s*y),     2 * (s*x + y*z), 1 - 2*(x**2 + y**2)]])
        
#     def Rotate(self, phi, theta, Phi):
#         y = self*Quaternion().E2Q(phi, theta, Phi)
#         return Quaternion(*list(y.reshape(-1)))
#     def dot_mul(self, other):
#         Q1 = self
#         Q2 = other
#         return Q1.s*Q2.s + Q1.x*Q2.x + Q1.y*Q2.y + Q1.z*Q2.z
        
    def mul(self,other):
        sa,xa,ya,za = self.s, self.x, self.y, self.z
        sb,xb,yb,zb = other.s, other.x,other.y,other.z
        s = sa*sb - xa*xb - ya*yb - za*zb
        x = sa*xb + sb*xa + ya*zb - yb*za
        y = sa*yb + sb*ya + za*xb - zb*xa
        z = sa*zb + sb*za + xa*yb - xb*ya
        return Quaternion(x,y,z,s)
        
    def inverse(self):
        return Quaternion(-self.x, -self.y, -self.z, self.s)
        
    def Rotate(self, vector):
        return vector.mul(self).mul(vector.inverse())
        
        
    def __add__(self, vector):
        return Quaternion(self.x+vector.x,self.y+vector.y,self.z+vector.z,self.s+vector.s)
    
    def __mul__(self, c):
        self.x, self.y, self.z, self.s = self.x*c, self.y*c, self.z*c, self.s*c
        return self
        
    def __repr__(self):
        return '({s},{x},{y},{z})--({phi},{theta},{Phi})'.format(s = self.s, x = self.x, y = self.y, z = self.z, phi = self.phi, theta = self.theta, Phi = self.Phi)

def getRotateQuaternion(Q1,Q2):
    cosTheta = (Q1.s*Q2.s + Q1.x*Q2.x + Q1.y*Q2.y + Q1.z*Q2.z)/(Q1.mean*Q2.mean)
    Theta = np.arccos(cosTheta)
    x = Q1.y*Q2.z - Q2.y*Q1.z
    y = Q1.z*Q2.x - Q2.z*Q1.x
    z = Q1.x*Q2.y - Q2.x*Q1.y
    mean = (x**2 + y**2 + z**2)**0.5
    vector = Quaternion(np.sin(Theta/2)*x/mean, np.sin(Theta/2)*y/mean, np.sin(Theta/2)*z/mean, np.cos(Theta/2))
    return vector


class SensorData:
    def __init__(self, ip = '192.168.1.108', port = 6666):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)#创建一个服务器socket监听
        self.sock.connect((self.ip, self.port))
        self.data = {}
        self.time = time.time()
        self.threading = threading.Thread(target=self.setData, args = (self,))
        self.threading.start()
        
    def getData(self):
        data = ''
        
        while True:
            b_data = self.sock.recv(256)
            data += b_data.decode('utf-8')
            try:
                jdata = json.loads(data.split('\n')[-2])
                data = ''
                break
            except:
                continue
        self.time = time.time()
        return jdata
    
    def setData(self, object):
        while True:
            self.data = self.getData()
            object.data = self.data
            
    def close(self):
        self.sock.close()
        
    def restart(self):
        self.__init__(self.ip, self.port)

myData = SensorData()
while True:
    try:
        t = myData.data['gravity']['timestamp']
        break
    except:
        pass

# ax = plt.subplot(121)
# ax1 = plt.subplot(122)
 

#初始化
s = (1 - np.linalg.norm(myData.data['rotationVector']['value'])**0.5)**0.5
r = Quaternion(*myData.data['rotationVector']['value'], s).inverse()

g = Quaternion(0,0,9.8)
o = Quaternion()
p1 = Quaternion(-5,-15,0).Rotate(r)
p2 = Quaternion(-5,15,0).Rotate(r)
p3 = Quaternion(5,15,0).Rotate(r)
p4 = Quaternion(5,-15,0).Rotate(r)

p5 = Quaternion(-5,-15,2).Rotate(r)
p6 = Quaternion(-5,15,2).Rotate(r)
p7 = Quaternion(5,15,2).Rotate(r)
p8 = Quaternion(5,-15,2).Rotate(r)

i = 0
x = []
g1 = []
g2 = []
g3 = []
la1 = []
la2 = []
la3 = []


fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')

while True:
    if myData.data['gravity']['timestamp'] - t > 0:
        i+=1
        x.append(i)
        g1.append(myData.data['gravity']['value'][0])
        g2.append(myData.data['gravity']['value'][1])
        g3.append(myData.data['gravity']['value'][2])
        la1.append(myData.data['linearAcceleration']['value'][0])
        la2.append(myData.data['linearAcceleration']['value'][1])
        la3.append(myData.data['linearAcceleration']['value'][2])

        gx = g1[-1]
        gy = g2[-1]
        gz = g3[-1]
        g_v = Quaternion(gx, gy, gz)
        r = getRotateQuaternion(g,g_v)
        s = (1 - np.linalg.norm(myData.data['rotationVector']['value'])**0.5)**0.5
        r = Quaternion(*myData.data['rotationVector']['value'], s)
        p1_ = p1.Rotate(r)
        p2_ = p2.Rotate(r)
        p3_ = p3.Rotate(r)
        p4_ = p4.Rotate(r)

        p5_ = p5.Rotate(r)
        p6_ = p6.Rotate(r)
        p7_ = p7.Rotate(r)
        p8_ = p8.Rotate(r)


        t = myData.data['gravity']['timestamp']
        print(myData.data)

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
        ax.set_xlim3d(-10,10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-10,10)

        plt.pause(0.01)
        # ax.clear()
        # ax.plot(x[-50:],g1[-50:],'r-')
        # ax.plot(x[-50:],g2[-50:],'g-')
        # ax.plot(x[-50:],g3[-50:],'b-')

        # ax1.clear()
        # ax1.plot(x[-50:],la1[-50:],'r-')
        # ax1.plot(x[-50:],la2[-50:],'g-')
        # ax1.plot(x[-50:],la3[-50:],'b-')

        # plt.show()







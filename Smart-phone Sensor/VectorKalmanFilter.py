import numpy as np

class VectorKalmanFilter:
    def __init__(self, s, M, A, B, H, Q, C):
        '''
        The number of observations is m; The number of signal estimates is p.
            s(n) = As(n-1) + Bu(n)
            x(n) = H(n)s(n) + w(n)
        The shape of input variables:
            s: (p,1) The estimates you want.
            A: (p,p) 
            B: (p,p) 
            M: (p,p) M(-1) = Cs, where s(-1) ~ N(mu_s, Cs)
            H: (m,p) 
            Q: (m,m) u(n) ~ N(0, Q)
            C: (p,p) w(n) ~ N(0, C)
        '''
        self.x_n_1 = None
        self.s_n_1 = s
        self.M_n_1 = M
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.C = C
    
    def forward(self, x_n):
        self.x_n_1 = x_n
        self.predict()
        self.s_n_1 = self.s_n_n_1 + self.K_n@(x_n - self.H@self.s_n_n_1)
        self.M_n_1 = (np.eye(self.K_n.shape[0]) - self.K_n@self.H)@self.M_n_n_1
        
        return self.s_n_1
    
    def predict(self):
        self.s_n_n_1 = self.A@self.s_n_1 # 1.prediction
        self.M_n_n_1 = self.A@self.M_n_1@self.A.T + self.B@self.Q@self.B.T # 2.Min prediction MSE
        self.K_n = self.M_n_n_1@self.H.T@np.linalg.inv(self.C + self.H@self.M_n_n_1@self.H.T) # 3.Kalman gain
    
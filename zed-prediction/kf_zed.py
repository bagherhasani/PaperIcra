from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt



class KF:
    def __init__(self,initial_px,initial_py,initial_vx,initial_vy,dt):

        self.initial_px=initial_px
        self.initial_py=initial_py
        self.initial_vx=initial_vx
        self.initial_vy=initial_vy
        self.dt=dt


        #kf initializing
        self.kf=KalmanFilter(4,2)  #4= px,py,vx,xy   2= px_zed, py_zed



        # define X state vector
        self.kf.x=np.array([self.initial_px,self.initial_py,self.initial_vx,self.initial_vy])



        # define F motion model
        """
        pxnew = px+(velx*dt)
        pynew = py+(vely*dt)
        vxnew = vx
        vynew = vy

        [px,py,vx,vy]        
        [1, 0 , dt, 0]
        [0, 1 , 0 , dt]
        [0, 0 , 1 , 0 ]
        [0, 0 , 0 , 1]
        """
        self.kf.F=np.array([[1 ,0  , self.dt,0],
                           [0 , 1 , 0 , self.dt],
                           [0 , 0 , 1 , 0],
                           [0 , 0 , 0 , 1]
                           ])
        


        #define H
        """
         measurement function here tell the kf that what parts can sensor measure
         my full state is x =[px,py,velx,vely]
         zed gives me only the [px,py]
         H connects state to measurement
        """
        self.kf.H=np.array([[1,0,0,0],
                            [0,1,0,0]])
        



        # Define P: initial uncertainty
        """
        P tells the Kalman filter how uncertain we are about the initial state.

        state = [px, py, vx, vy]

        px, py:
            position from ZED is  somehow trusted

        vx, vy:
            initial velocity is more uncertain because at the beginning
            we may not know how fast the person is moving yet
        """
        self.kf.P = np.array([
            [1,   0,   0,   0],
            [0,   1,   0,   0],
            [0,   0, 100,   0],
            [0,   0,   0, 100]
        ])

        


        """
        Define R: Reasurement noise - how noisy the zed-camera might be ?
        measurements are [zedpx, zed py] 
        """
        
        self.kf.R= np.array([[0.1 , 0]
                            , [0 , 0.1]])
        


        """Define Q: Process noise / motion model uncertainty

            Meaning: how imperfect my motion model is.

            My motion model assumes constant velocity:
            px_new = px + vx * dt
            py_new = py + vy * dt
            vx_new = vx
            vy_new = vy

            But people do not move with perfect constant velocity.
            They can slow down, speed up, or turn.

            Q allows the Kalman filter to accept that the motion model is imperfect.
            Small Q  -> trust constant velocity more
            Large Q  -> allow motion to change more
            """

        self.kf.Q = np.array([
                            [0.01, 0,    0,   0],
                            [0,    0.01, 0,   0],
                            [0,    0,    0.1, 0],
                            [0,    0,    0,   0.1]
                        ])


    def updateF(self, dt):
        self.dt=dt

        self.kf.F=np.array([
            [1,0,self.dt,0],
            [0,1,0, self.dt],
            [0, 0, 1, 0],
            [0,0,0,1]
        ])


    def preidt(self):
        self.kf.predict()

        return self.kf.x
    

    def update(self,measured_x,measured_y):
        self.kf.update(np.array([measured_x,measured_y]))

        return self.kf.x
    

    def processMeasurement(self,measured_posx,measured_posy,dt):
        self.updateF(dt)
        self.preidt()
        self.update(measured_posx,measured_posy)

        px=float(self.kf.x.flatten()[0])
        py=float(self.kf.x.flatten()[1])
        vx=float(self.kf.x.flatten()[2])
        vy=float(self.kf.x.flatten()[3])

        return px,py, vx, vy
    


    def predictFuture(self,seconds_ahead):
        
        
        px=float(self.kf.x.flatten()[0])
        py=float(self.kf.x.flatten()[1])
        vx=float(self.kf.x.flatten()[2])
        vy=float(self.kf.x.flatten()[3])

        future_px=px+(vx*seconds_ahead)
        future_py=py+(vy*seconds_ahead)

        return future_px,future_py
    

        



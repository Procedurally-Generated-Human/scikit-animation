from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.inspection import DecisionBoundaryDisplay
from .animator import Animator


class SGDClassifierAnimator(Animator):
    

    def __animation_init(self):
        self.ax.set_title("Degree "+str(self.deg)+" Polynomial Classification")

    def __animation_update(self):
        self.ax.scatter(self.x[:, 0], self.y, color='red')
        for i in range(10):
            self.model.partial_fit(self.scaled_x,self.y)
        myline = np.linspace(np.amin(self.x[:, 0]), np.amax(self.x[:, 0]), 100).reshape(-1,1)
        myline_scaled = preprocessing.PolynomialFeatures(degree=self.deg, include_bias=False).fit_transform(myline)
        myline_scaled = preprocessing.StandardScaler().fit(self.x).transform(myline_scaled)
        preds = self.model.predict(myline_scaled)
        plt.plot(myline, preds, color='blue')
        self.camera.snap()


    def animate(self):
        self.__animation_init()
        for i in range(1000):
            self.__animation_update()
        self.animation = self.camera.animate(interval = 1, repeat = True,
                           repeat_delay = 500)
        plt.show()

    
    def save(self, name:str, format:str="mp4"):
        self.__animation_init()
        for i in range(1000):
            self.__animation_update()
        animation = self.camera.animate(interval = 40, repeat = True,
                           repeat_delay = 500)
        
        filename = name + "." + format
        print("Creating animation...")
        animation.save(filename)
        print("Animation Saved")




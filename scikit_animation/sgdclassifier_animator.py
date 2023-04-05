from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.inspection import DecisionBoundaryDisplay
from .animator import Animator


class SGDClassifierAnimator(Animator):
    

    def __animation_init(self):
        self.ax.set_title("Degree "+str(self.deg)+" Polynomial Classification")
        self.score = 0.0

    def __animation_update(self):
        if self.score == 1.0:
            return
          
        labels = np.unique(self.y)
        self.model.partial_fit(self.x, self.y, labels)
        self.score = self.model.score(self.x, self.y)
        

        
        self.dict = dict.fromkeys(labels)
        for val in self.dict:
            self.dict[val] = []
        for i in range(0,self.x.shape[0]):
            self.dict[self.y[i]].append(self.x[i])
        colors = list("rgbcmyk")
        for x in self.dict.values():
            x = np.array(x)
            plt.scatter(x[:, 0],x[:, 1],color=colors.pop())
        
        
        
        h = .02  # step size in the mesh

        x_min, x_max = self.x[:, 0].min() - 1, self.x[:, 0].max() + 1
        y_min, y_max = self.x[:, 1].min() - 1, self.x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        
        self.camera.snap()


    def animate(self):
        self.__animation_init()
        for i in range(100):
            self.__animation_update()
        self.animation = self.camera.animate(interval = 100, repeat = False,
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





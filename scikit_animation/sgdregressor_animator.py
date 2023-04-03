from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from .animator import Animator, Animator2
from celluloid import Camera



class SGDRegressorAnimator(Animator):
    
    def animation_init(self):
        self.line.set_data(self.x,self.y)
        self.ax.scatter(self.x[:, 0], self.y, color='orange', label="Target")
        self.ax.set_title(str(self.deg) + " Poly Regression")
        return self.line,

    def animation_update(self, i):
        self.model.partial_fit(self.scaled_x,self.y)
        myline = np.linspace(np.amin(self.x[:, 0]), np.amax(self.x[:, 0]), 1000)
        poly = PolynomialFeatures(degree=self.deg, include_bias=False)
        poly_features1 = poly.fit_transform(myline.reshape(-1, 1))
        scaler = preprocessing.StandardScaler().fit(poly_features1)
        poly_features1 = scaler.transform(poly_features1)
        pred = self.model.predict(poly_features1)
        self.line.set_data(myline, pred)
        return self.line,



class SGDRegressorAnimator2(Animator2):
    
    def __animation_init(self):
        self.ax.set_title(str(self.deg) + " Poly Regression")

    def __animation_update(self):
        self.ax.scatter(self.x,self.y, color='red')
        for i in range(10):
            self.model.partial_fit(self.x,self.y)
        myline = np.linspace(np.amin(self.x[:, 0]), np.amax(self.x[:, 0]), 10).reshape(-1,1)
        preds = self.model.predict(myline)
        plt.plot(myline, preds, color='blue')
        self.camera.snap()


    def animate(self):
        self.__animation_init()
        for i in range(1000):
            self.__animation_update()
        animation = self.camera.animate(interval = 1, repeat = False,
                           repeat_delay = 500)
        plt.show()

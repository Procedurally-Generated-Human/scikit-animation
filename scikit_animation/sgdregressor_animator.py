from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from .animator import Animator


class SGDRegressorAnimator(Animator):
    

    def __animation_init(self):
        self.ax.set_title("Degree "+str(self.deg)+" Polynomial Regression")

    def __animation_update(self):
        self.ax.scatter(self.x[:, 0], self.y, color='red')
        for i in range(10):
            self.model.partial_fit(self.scaled_x,self.y)
        myline = np.linspace(np.amin(self.x[:, 0]), np.amax(self.x[:, 0]), 100).reshape(-1,1)
        myline_scaled = preprocessing.PolynomialFeatures(degree=self.deg, include_bias=False).fit_transform(myline)
        myline_scaled = preprocessing.StandardScaler().fit(self.x).transform(myline_scaled)
        preds = self.model.predict(myline_scaled)
        plt.plot(myline, preds, color='blue',)
        self.camera.snap()


    def __animation_init_wc(self):
        self.ax[0].set_title("Degree "+str(self.deg)+" Polynomial Regression")
        self.ax[1].set_title("Learning Curve")
        self.ax[1].set_xlabel("Iteration")
        self.ax[1].set_ylabel("Score")
        self.ax[1].set_ylim([0, 1])



    def __animation_update_wc(self):
        self.ax[0].scatter(self.x[:, 0], self.y, color='red')
        for i in range(10):
            self.model.partial_fit(self.scaled_x,self.y)
        myline = np.linspace(np.amin(self.x[:, 0]), np.amax(self.x[:, 0]), 100).reshape(-1,1)
        myline_scaled = preprocessing.PolynomialFeatures(degree=self.deg, include_bias=False).fit_transform(myline)
        myline_scaled = preprocessing.StandardScaler().fit(self.x).transform(myline_scaled)
        preds = self.model.predict(myline_scaled)
        self.ax[0].plot(myline, preds, color='blue')

        score = self.model.score(self.scaled_x, self.y)
        self.score_list.append(score)
        self.counter += 1
        self.ax[1].plot(list(range(self.counter)), self.score_list, color="blue")
        self.camera.snap()


    def animate(self):
        if self.animate_cost:
            self.__animation_init_wc()
            for i in range(500):
                self.__animation_update_wc()
            self.animation = self.camera.animate(interval = 1, repeat = True,
                           repeat_delay = 500)
            plt.show()
            
        else:
            self.__animation_init()
            for i in range(1000):
                self.__animation_update()
            self.animation = self.camera.animate(interval = 1, repeat = True,
                           repeat_delay = 500)
            plt.show()

    
    def save(self, name:str, format:str="mp4"):

        if self.animate_cost:
            self.__animation_init_wc()
            for i in range(500):
                self.__animation_update_wc()
            self.animation = self.camera.animate(interval = 20, repeat = True,
                           repeat_delay = 500)
            
        else:
            self.__animation_init()
            for i in range(1000):
                self.__animation_update()
            self.animation = self.camera.animate(interval = 1, repeat = True,
                           repeat_delay = 500)


        filename = name + "." + format
        print("Creating animation...")
        self.animation.save(filename)
        print("Animation Saved")
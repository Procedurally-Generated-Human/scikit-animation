from matplotlib import animation, pyplot as plt
import numpy as np
from sklearn import preprocessing


class Animator:

    def __init__(self, model, x, y, deg=1):
        self.model = model
        self.x = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x.reshape(-1, 1))
        self.y = y
        self.deg = deg
        self.scaled_x = preprocessing.StandardScaler().fit(self.x).transform(self.x)
    
    def animate(self):
        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(self.x[:, 0], self.y, color='orange', label="Target")
        self.line, = ax.plot([], [], lw=3, label="Estimate")
        fig.legend(loc="upper left")
        ax.set_title(str(self.deg) + " Poly Regression")
        self.anim = animation.FuncAnimation(fig, self.animation_update, init_func=self.animation_init,
                                frames=600, interval=10, blit=True)
        plt.show()

    def animation_init(self):
        self.line.set_data(self.x,self.y)
        return self.line,

    def animation_update(self, i):
        pass


    def save(self, name, fps):
        pass
    



    
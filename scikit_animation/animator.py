from matplotlib import animation, pyplot as plt
import numpy as np
from sklearn import preprocessing


class Animator:

    def __init__(self, model, x, y, deg=1):
        self.model = model
        self.model_copy = model
        self.x = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x.reshape(-1, 1))
        self.XX = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x)
        self.y = y
        self.deg = deg
        self.scaled_x = preprocessing.StandardScaler().fit(self.x).transform(self.x)
    
    def animate(self):
        fig = plt.figure()
        self.ax = plt.axes()
        self.line, = self.ax.plot([], [], lw=3, label="Estimate")
        fig.legend(loc="upper left")
        self.animation_init()
        self.anim = animation.FuncAnimation(fig, self.animation_update,
                                frames=600, interval=10, blit=True)
        plt.show()

    def animation_init(self):
        self.line.set_data(self.x,self.y)
        return self.line,

    def animation_update(self, i):
        return self.line,


    def save(self, name, fps):
        self.model = self.model_copy
        print("Creating Video...")
        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(self.x[:, 0], self.y, color="orange" ,label="Target")
        fig.legend(loc="upper left")
        ax.set_title(str(self.deg) + " Poly Regression")
        self.line, = ax.plot([], [], lw=2, label="Estimate")
        self.anim = animation.FuncAnimation(fig, self.animation_update, init_func=self.animation_init,
                                frames=200, interval=100, blit=True)
        writer = animation.FFMpegWriter(
        fps=fps, metadata=dict(artist='Parsa Toopchinezhad'), bitrate=100)
        self.anim.save(name, writer=writer)
        print("Video saved as", name)
    

    



    
from matplotlib import animation, pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from animator import Animator


class SGDRegressorAnimator(Animator):
    
    

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

    def save(self, name, fps):
        print("Creating Video...")
        self.model = SGDRegressor()
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

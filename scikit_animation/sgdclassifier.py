from matplotlib import pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from .animator import Animator


class SGDClassifierAnimator(Animator):
    
    def animation_init(self):
        labels = np.unique(self.y)
        self.dict = dict.fromkeys(labels)
        for val in self.dict:
            self.dict[val] = []
        
        for i in range(0,self.XX.shape[0]):
            self.dict[self.y[i]].append(self.XX[i])
        self.ax.plot([row[0] for row in self.dict[1]], [row[1] for row in self.dict[1]],"o", [row[0] for row in self.dict[2]], [row[1] for row in self.dict[2]], "x")
    

    def animation_update(self, i):
        self.model.partial_fit(self.XX,self.y, classes=np.unique(self.y))
        tmp = self.model.coef_[0]
        a = - tmp[0] / tmp[1]

        b = - (self.model.intercept_[0]) / tmp[1]

        xx = np.linspace(np.amin(self.XX[:, 0]), np.amax(self.XX[:, 0]),10000)
        yy = a * xx + b
        self.line.set_data(xx, yy)
        print(self.model.score(self.XX, self.y))
        return self.line,



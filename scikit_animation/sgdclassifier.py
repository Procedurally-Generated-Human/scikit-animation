from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from .animator import Animator


class SGDClassifierAnimator(Animator):
    
    def animation_init(self):
        labels = np.unique(self.y)
        self.dict = dict.fromkeys(labels)
        for val in self.dict:
            self.dict[val] = []
        print(self.x.shape)
        
        for i in range(0,self.XX.shape[0]):
            self.dict[self.y[i]].append(self.XX[i])
        print(self.dict)
        self.ax.plot([row[0] for row in self.dict[1]], [row[1] for row in self.dict[1]],"o", [row[0] for row in self.dict[2]], [row[1] for row in self.dict[2]], "x")

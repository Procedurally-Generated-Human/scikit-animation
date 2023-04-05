from matplotlib import pyplot as plt
from sklearn import preprocessing
from celluloid import Camera



class Animator:

    def __init__(self, model, x, y, deg=1, animate_cost=False) -> None:
        self.model = model
        self.model_copy = model
        self.x = x
        self.scaled_x = preprocessing.StandardScaler().fit(self.x).transform(self.x)
        self.y = y
        self.deg = deg
        self.fig, self.ax = plt.subplots()
        self.animate_cost = animate_cost
        if animate_cost:
            self.fig, self.ax = plt.subplots(2, 1)
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=0.3)
        self.camera = Camera(self.fig)
        self.counter = 0
        self.score_list = []

    def __animation_init(self):
        pass

    def __animation_update(self):
        pass
        

    def animate(self):
        pass


    def save(self, name:str, format:str="mp4"):
        pass
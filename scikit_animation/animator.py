from matplotlib import pyplot as plt
from sklearn import preprocessing
from celluloid import Camera



class Animator:

    def __init__(self, model, x, y, deg=1, speed=1) -> None:
        self.model = model
        self.model_copy = model
        self.x = x
        self.scaled_x = preprocessing.StandardScaler().fit(self.x).transform(self.x)
        self.y = y
        self.deg = deg
        self.speed = speed
        self.fig, self.ax = plt.subplots()
        self.camera = Camera(self.fig)

    def __animation_init(self):
        pass

    def __animation_update(self):
        pass
        

    def animate(self):
        pass


    def save(self, name:str, format:str="mp4"):
        pass
# Scikit-Animation
A python library for producing animated visuals from scikit-learn models. Intended for edjucational purposes.  
</br>
![1 poly regression](https://github.com/Procedurally-Generated-Human/scikit-animation/blob/main/videos/10.gif)

## Installation
Clone the project repository, or install via pip:
```
pip install scikit-animation
```

## Using the Library
For this example, we will be animating Scikit-learn's 'SGDRegressor' model which uses gradient decent:
``` python
import numpy as np
from sklearn.linear_model import SGDRegressor
from scikit_animation.sgdregressor_animator import SGDRegressorAnimator


# create x and y values
x  = np.array([0,2,4,6,8,10,12,14,16]).reshape(-1,1)
y = np.array( [0,1,4,9,16,25,30,25,16])

# add polynomial features to x
x = preprocessing.PolynomialFeatures(degree=4, include_bias=False).fit_transform(x)

# create the scikit-learn model
model = SGDRegressor(penalty=None)

# create the animator model and show the animation
animator = SGDRegressorAnimator(model, x, y, deg=4, animate_cost=True) # setting animate_cost to True will animate the cost function
animator.animate()

# optional: save the animation
i.save("10-degree-poly","mp4")

```

</br>
Note: this library currently only animates SGDRegressor. more models will be added in the future

import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import time
import random
from urllib.request import proxy_bypass
from math import pi



import numpy as np

def entropy(probabilities):
    # Ensure that probabilities sum up to 1
    assert np.isclose(np.sum(probabilities), 1.0)

    # Convert generator to NumPy array
    probabilities_array = np.fromiter((p * np.log2(p) if p != 0 else 0 for p in probabilities), float)

    # Compute the entropy
    entropy = - np.sum(probabilities_array)

    return entropy

# Example usage:
probabilities = [0.3, 0.4, 0.2, 0.1]
result = entropy(probabilities)
print(result)


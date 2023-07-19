import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import time
import random
from urllib.request import proxy_bypass
from math import pi







def visualize_weights(model):
    # Extract the weights from the model
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy())

    # Plot the weights
    fig, axs = plt.subplots(len(weights), 1, figsize=(8, 6))
    for i, weight in enumerate(weights):
        if len(weight.shape) > 1:
            axs[i].imshow(weight.squeeze(), cmap='gray')
        else:
            axs[i].plot(weight)
        axs[i].set_title(f'Layer {i + 1} weights')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.pause(10)


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate random data for 10 lists of mean scores
# num_lists = 10
# num_scores = 100
# mean_scores = []
# scores = []
# for _ in range(num_lists):
#     scores.append(list(np.random.rand(num_scores)))
#
#
# for n in range(num_lists):
#     for i in range(100):
#         scores[n][i]*=5*i
#
#
# mean_scores =  np.mean(scores,axis=0)
#
# # Plotting parameters
# alpha_value = 0.2  # Alpha value for faded background
# average_color = 'blue'  # Color for the average line
#
# # Plot each list of mean scores with a faded background
# for i, score in enumerate(scores):
#     plt.plot(score, color='orange', alpha=alpha_value)
#
# # Plot the average line with a solid color
# # average_scores = np.mean(mean_scores, axis=0)
# plt.plot(mean_scores, color=average_color, label='Average')
#
# # Add labels and title
# plt.xlabel('Iterations')
# plt.ylabel('Mean Score')
# plt.title('Average of Multiple Lists of Mean Scores')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.show()


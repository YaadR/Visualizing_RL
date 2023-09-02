import matplotlib.pyplot as plt
from IPython import display
from scipy.ndimage import gaussian_filter
import numpy as np

plt.ion()


def plot(scores, mean_scores, save=False, index=100):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 3)))
    plt.show(block=False)
    plt.pause(0.1)
    if save:
        plt.savefig("edvance_" + str(index) + ".jpg")
        plt.close()


def heat_map_step(
    matrix, direction, W, H, head_x, head_y, danger_ahead=False, X=0, Y=0
):
    # Power parameters
    ADD = 1.01
    SUB = 0.99
    DANGER = 0.95

    # Look ahead - closing bounderies
    if direction == 1:  # Right
        matrix[head_x:] *= ADD
        matrix[:head_x] *= SUB

    elif direction == 2:  # Left
        matrix[:head_x] *= ADD
        matrix[head_x:] *= SUB
    elif direction == 3:  # Up
        matrix[:, :head_y] *= ADD
        matrix[:, head_y:] *= SUB
    else:  # Down
        matrix[:, head_y:] *= ADD
        matrix[:, :head_y] *= SUB

    # Approximation of probable reward exact location
    est_rew_x = head_x + X
    est_rew_y = head_y + Y
    if (est_rew_x >= 0 and est_rew_x < W) and (est_rew_y >= 0 and est_rew_y < H):
        matrix[est_rew_x, est_rew_y] *= 1.05

    # Danger ahead
    elif danger_ahead:
        matrix[head_x, head_y] *= DANGER

    # Returned value after gaussian smoothing
    return gaussian_filter(matrix, sigma=0.5)


def distance_collapse(distance, w, h, dir):
    X = int(distance[0] * w)
    if dir == 2:  # Left
        X *= -1
    Y = int(distance[1] * h)
    if dir == 3:
        Y *= -1
    return X, Y


def normalize(array):
    min_value = min(array)
    max_value = max(array)
    if min_value == max_value:
        return array
    return [(value - min_value) / (max_value - min_value) for value in array]


def normalizer(values):
    total_sum = sum(values)
    normalized_values = [value / total_sum for value in values]
    return normalized_values


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def net_visualize(model, axs):
    # Extract the biases from the model
    layers = []
    for param in model.parameters():
        layers.append(param.data.cpu().numpy())
    # Plot the biases
    for i, layer in enumerate(layers):
        if len(layer.shape) == 1:
            bias_img = np.reshape(layer, (-1, len(layer)))
            if len(layer) > 32:
                bias_img = np.reshape(
                    bias_img, (int(np.sqrt(len(layer))), int(np.sqrt(len(layer))))
                )
            else:
                bias_img = bias_img.reshape(-1, 1)
            axs[i].imshow(bias_img, cmap="gray")
            axs[i].set_title(f"Biases {(i + 1)//2} ")
        else:
            if layer.shape[0] > layer.shape[1]:
                layer = layer.T

            axs[i].imshow(layer.T, cmap="gray")
            axs[i].set_title(f"Wights {i + 1}")

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # return last_bias


def weight_visualize(model, axs):
    # Extract the biases from the model
    layers = []
    for param in model.parameters():
        layers.append(param.data.cpu().numpy())
    # Plot the biases
    for i, layer in enumerate(layers):
        if len(layer.shape) == 1:
            weight_layer = np.reshape(layer, (-1, 1))

            if len(weight_layer) > 32:
                layer_widen = np.zeros((len(weight_layer), 100))
                for d, v in enumerate(weight_layer):
                    layer_widen[d] = v[0]

                weight_layer = layer_widen
            axs[i // 2].imshow(weight_layer, cmap="viridis")
            axs[i // 2].set_title(f"weight layer {i//2}")
    for g in range(len(axs)):
        axs[g].set_xticks([])
        axs[g].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    plt.show()
    plt.pause(0.1)

    # return last_bias


def activation_visualize(
    state_vector, layer1, layer2, axs, snapshot, index="", activation_name="Valid State"
):
    # Extract the weights from the model
    layer_widen = np.zeros((len(layer1.T), 50))
    for i, v in enumerate(layer1.T):
        layer_widen[i] = v

    axs[0].imshow(state_vector.T, cmap="viridis")
    axs[0].set_title(activation_name)

    axs[1].imshow(layer_widen, cmap="viridis")
    axs[1].set_title("Layer 1")

    axs[2].imshow(layer2.T, cmap="viridis")
    axs[2].set_title("Layer 2")

    axs[3].imshow(snapshot)

    plt.tight_layout()
    plt.savefig(
        f"D:\GitHub\Reposetories\Visualizing_RL\Ver1\data\plots\\valid activation\state_{index}.jpg"
    )  # Specify the desired file name and extension
    plt.show()

    plt.pause(2)
    # return last_bias


def array_tobinary(state):
    sb = ""
    for i in state:
        sb += str(i)
    return int(sb, 2)


def plot_mean_scores_buffer(scores, title):
    min_length = min(len(sublist) for sublist in scores)

    cut_list = [sublist[:min_length] for sublist in scores]
    padded_list = [
        sublist + [np.nan] * (min_length - len(sublist)) for sublist in cut_list
    ]
    mean_list = np.nanmean(padded_list, axis=0)

    # Plotting parameters
    alpha_value = 0.2  # Alpha value for faded background
    average_color = "purple"  # Color for the average line

    # Plot each list of mean scores with a faded background
    for i, score in enumerate(scores):
        plt.plot(score, color="green", alpha=alpha_value)

    # Plot the average line with a solid color
    # average_scores = np.mean(mean_scores, axis=0)
    plt.plot(mean_list, color=average_color, label="Mean")

    # Add labels and title
    plt.xlabel("Iterations")
    plt.ylabel("Mean Score")
    plt.title("Multiple Training Mean " + title)

    # Add legend
    plt.legend()

    # Save fig
    plt.savefig(
        r"D:\GitHub\Reposetories\Visualizing_RL\Ver1\data\plots\Buffers\Mean "
        + title
        + ".jpg"
    )  # Specify the desired file name and extension

    # Show the plot
    plt.show()


def plot_std_mean_scores_buffer(mean_scores, title):
    data_std = np.std(mean_scores, axis=0)
    data_mean = np.mean(mean_scores, axis=0)
    plt.figure(figsize=(8, 5))
    plt.title("Multiple Training STD " + title)

    alpha = 0.2

    plt.plot(range(len(data_mean)), data_mean, "-", color="r", label="Mean")
    plt.fill_between(
        range(len(data_mean)),
        data_mean - data_std,
        data_mean + data_std,
        facecolor="b",
        alpha=alpha,
    )

    plt.xlabel("Games")
    plt.ylabel("Mean Score")

    plt.savefig(
        r"D:\GitHub\Reposetories\Visualizing_RL\Ver1\data\plots\Buffers\STD "
        + title
        + ".jpg"
    )  # Specify the desired file name and extension
    plt.legend()
    plt.tight_layout()
    plt.show()


def entropy(probabilities):
    # Ensure that probabilities sum up to 1
    assert np.isclose(np.sum(probabilities), 1.0)

    # Convert generator to NumPy array
    probabilities_array = np.fromiter(
        (p * np.log2(p) if p != 0 else 0 for p in probabilities), float
    )

    # Compute the entropy
    entropy = -np.sum(probabilities_array)

    return entropy


def plot_system_entropy(entropies):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # plt.clf()
    plt.title("System Entropy")
    plt.xlabel("Games")
    plt.ylabel("Entropy")
    plt.plot(entropies)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(0.1)


def cirtenty_function(certainty):
    # the decision space is 1 of 3, thus maximal entropy is log(1/3)
    return -np.log2(1 / 3) - certainty

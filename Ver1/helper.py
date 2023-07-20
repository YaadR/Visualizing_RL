import matplotlib.pyplot as plt
from IPython import display
from scipy.ndimage import gaussian_filter
import numpy as np

plt.ion()

def plot(scores, mean_scores, save=False,index=100):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    if save:
        plt.savefig('edvance_'+str(index)+'.jpg')
        plt.close()

def plot_multi(mean_scores, maxmean = 5):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # plt.clf()
    plt.title('Training')
    plt.xlabel('Number of Games')
    plt.ylabel('Mean Score')
    # plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    # plt.text(len(scores)-1, max(scores), str(max(scores)))
    plt.text(len(mean_scores)-1, maxmean, str(maxmean))
    plt.show(block=False)
    plt.pause(.1)


def heat_map_step(matrix,direction,W,H,head_x,head_y, danger_ahead=False,X=0,Y=0):

    # Power parameters
    ADD = 1.01
    SUB = 0.99
    DANGER = 0.95

    # Look ahead - closing bounderies
    if direction ==1:           # Right
        matrix[head_x:] *= ADD
        matrix[:head_x] *= SUB

    elif direction ==2:         # Left
        matrix[:head_x] *= ADD
        matrix[head_x:] *= SUB
    elif direction ==3:         # Up
        matrix[:,:head_y] *= ADD
        matrix[:,head_y:] *= SUB
    else:                       # Down
        matrix[:,head_y:] *= ADD
        matrix[:,:head_y] *= SUB

    # Approximation of probable reward exact location
    est_rew_x = head_x + X
    est_rew_y = head_y + Y
    if (est_rew_x >=0 and est_rew_x< W) and (est_rew_y >=0 and est_rew_y < H):
        matrix[est_rew_x,est_rew_y] *=1.05

    # Danger ahead
    elif danger_ahead:
        matrix[head_x,head_y] *= DANGER

    # Returned value after gaussian smoothing
    return gaussian_filter(matrix, sigma=0.5)

def distance_collapse(distance,w,h,dir):
    X = int(distance[0]*w)
    if dir == 2:        #Left
        X *=-1
    Y = int(distance[1]*h)
    if dir==3:
        Y *=-1
    return  X,Y

def normalize(array):
    min_value = min(array)
    max_value = max(array)
    if min_value==max_value:
        return array
    return [(value - min_value) / (max_value - min_value) for value in array]

def normalizer(values):
    total_sum = sum(values)
    normalized_values = [value / total_sum for value in values]
    return normalized_values

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

def visualize_biases(model,axs,last_bias,diff,loss,epsilon_decay=None,agent_type=1,loss_1=None,):
    # Extract the biases from the model
    biases = []
    j=-1
    for param in model.parameters():
        biases.append(param.data.cpu().numpy())
    # Plot the biases
    for i, bias in enumerate(biases):
        if len(bias.shape)==1:
            j += 1
            bias_img = np.reshape(bias, (-1, len(bias)))
            if len(bias)>32:

                bias_img = np.reshape(bias_img,(int(np.sqrt(len(bias))),int(np.sqrt(len(bias)))))
                if j==0:
                    diff.append(np.mean(np.abs(bias_img-last_bias)))
                    last_bias = bias_img.copy()
            else:
                bias_img = bias_img.reshape(-1,1)
            axs[j].imshow(bias_img, cmap='gray')
            axs[j].set_title(f'Layer {j + 1} biases')
            # axs[j].axis('off')
    axs[j + 1].clear()
    loss = normalize(loss)
    # print(loss)
    axs[j + 1].plot(loss, 'orange')
    diff = normalize(diff)
    # print(diff[-1])
    axs[j + 1].plot(diff, 'b')
    # loss_1 = normalize(loss_1)
    # axs[j+1].plot(loss_1,'b')
    if agent_type:
        if max(epsilon_decay)>1:
            epsilon_decay = normalize(epsilon_decay)
        # epsilon_decay[-1] = 0 if epsilon_decay[-1]<0 else epsilon_decay[-1]
        axs[j + 1].plot(epsilon_decay, 'pink')
    axs[j + 1].set_title('Mean layer 1 difference')

    for g in range(len(axs)-1):
        axs[g].set_xticks([])
        axs[g].set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return last_bias


def net_visualize(model,axs):
    # Extract the biases from the model
    layers = []
    for param in model.parameters():
        layers.append(param.data.cpu().numpy())
    # Plot the biases
    for i, bias in enumerate(layers):
        if len(bias.shape)==1:
            bias_img = np.reshape(bias, (-1, len(bias)))
            if len(bias)>32:
                bias_img = np.reshape(bias_img,(int(np.sqrt(len(bias))),int(np.sqrt(len(bias)))))
            else:
                bias_img = bias_img.reshape(-1,1)
            axs[i].imshow(bias_img, cmap='gray')
            axs[i].set_title(f'Biases {(i + 1)//2} ')
            # axs[j].axis('off')
        else:
            if bias.shape[0] > bias.shape[1]:
                bias = bias.T
            # if i==0:
            #     bias_img = np.reshape(bias, (16*4,13*4))
            # elif i==2:
            #     bias_img = np.reshape(bias, (16*2, 3*8))
            axs[i].imshow(bias.T, cmap='gray')
            axs[i].set_title(f'Wights {i + 1}')
    # for g in range(len(axs)):
    #     axs[g].set_xticks([])
    #     axs[g].set_yticks([])


    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # return last_bias

def table_visualize(table,axs,mean_score,plot_scores):
    # Extract the biases from the model

    layers = []
    axs[0].imshow(table, cmap='gray')
    axs[0].set_title("Q-Table")

    axs[1].plot(plot_scores, 'b')
    axs[1].plot(mean_score, 'orange')
    axs[1].set_title('Scores')

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # return last_bias


def activation_visualize(state_vector,layer1,layer2,axs,index=0,activation_name='Valid State Activation'):
    # Extract the biases from the model
    layer_widen = np.zeros((len(layer1.T),50))
    for i,v in enumerate(layer1.T):
        layer_widen[i] = v

    axs[0].imshow(state_vector.T, cmap='viridis')
    axs[0].set_title(activation_name)


    axs[1].imshow(layer_widen, cmap='viridis')
    axs[1].set_title('Layer 1 activation')

    axs[2].imshow(layer2.T, cmap='viridis')
    axs[2].set_title('Layer 2 activation')

    plt.tight_layout()
    plt.savefig(f'sample_state{index}.jpg')  # Specify the desired file name and extension
    plt.show()

    plt.pause(2)
    # return last_bias

def array_tobinary(state):
    sb = ""
    for i in state:
        sb += str(i)
    return int(sb, 2)


def plot_mean_scores_buffer(scores):
    min_length = min(len(sublist) for sublist in scores)

    cut_list = [sublist[:min_length] for sublist in scores]
    padded_list = [sublist + [np.nan] * (min_length - len(sublist)) for sublist in cut_list]
    mean_list = np.nanmean(padded_list, axis=0)
    # for n in range(num_lists):
    #     for i in range(100):
    #         scores[n][i] *= 5 * i

    #mean_scores = np.mean(scores, axis=0)

    # Plotting parameters
    alpha_value = 0.2  # Alpha value for faded background
    average_color = 'purple'  # Color for the average line

    # Plot each list of mean scores with a faded background
    for i, score in enumerate(scores):
        plt.plot(score, color='green', alpha=alpha_value)

    # Plot the average line with a solid color
    # average_scores = np.mean(mean_scores, axis=0)
    plt.plot(mean_list, color=average_color, label='Average')

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Mean Score')
    plt.title('Average of Multiple Lists of Mean Scores')

    # Add legend
    plt.legend()

    #Save fig
    plt.savefig(r'D:\GitHub\Reposetories\Visualizing_RL\Ver1\data\plots\Buffers\buffer.jpg')  # Specify the desired file name and extension

    # Show the plot
    plt.show()

def plot_std_mean_scores_buffer(data_mean, data_std, data_label, x_label, y_label, title):

    plt.figure(figsize=(8,5))
    plt.title(title)

    alpha = 0.3

    plt.plot(range(len(data_mean)), data_mean, '-', color = 'purple', label = data_label)
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, facecolor = 'blue',alpha = alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()
    plt.tight_layout()
    plt.show()
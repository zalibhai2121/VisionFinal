import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_and_visualize_model(filename):
    np.set_printoptions(precision=3, suppress=True)
    model = tf.keras.models.load_model(filename)
    v = tf.squeeze(model.trainable_variables[0]).numpy()
    num_features = v.shape[2]
    fig, ax = plt.subplots(nrows=4, ncols=num_features//4)
    ax = ax.flatten()
    for i in range(num_features):
        img = v[:,:,i]
        ax[i].imshow(img, cmap='Greys')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        print(img)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename + ".png")

load_and_visualize_model('models/asl_1')

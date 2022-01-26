from matplotlib.pyplot import sca
import tensorflow as tf
import utils

def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    logits = tf.exp(logits)
    denom = tf.reduce_sum(logits)
    soft_logits = logits / denom
    
    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION
    y_hat = tf.boolean_mask(scaled_logits, one_hot, axis=0)
    log_y_hat = tf.math.log(y_hat)
    loss = - log_y_hat
    
    return loss


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
    X = tf.reshape(X, [1, -1])
    y_hat = softmax(tf.matmul(X, W) + b)
    
    return y_hat


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - Y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # IMPLEMENT THIS FUNCTION
    class_idx = tf.argmax(y_hat, axis=1, output_type=tf.int32)
    correct = tf.cast(tf.equal(class_idx, Y), tf.int32)
    correct = tf.reduce_sum(correct)
    acc = correct / Y.shape[0]

    return acc


if __name__ == "__main__":
    utils.check_softmax(softmax)
    utils.check_ce(cross_entropy)
    utils.check_model(model)
    utils.check_acc(accuracy)
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import argparse
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_reg):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    stddev = 0.01
    input = tf.layers.conv2d(
        inputs=vgg_layer7_out, 
        filters=num_classes,
        kernel_size=1, 
        strides=(1,1),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "inception")
    

    # beginning of the decoder: de-convolutional layer
    input = tf.layers.conv2d_transpose(
        inputs=input, 
        filters=num_classes,
        kernel_size=4, 
        strides=(2,2),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "decoder_layer1")

    pool4_upscale = tf.layers.conv2d(
        inputs=vgg_layer4_out, 
        filters=num_classes,
        kernel_size=1, 
        strides=(1,1),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "score_pool4")

    input = tf.add(input, pool4_upscale, name="skip_pool4")

    input = tf.layers.conv2d_transpose(
        inputs=input, 
        filters=num_classes,
        kernel_size=4, 
        strides=(2,2),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "decoder_layer2")

    pool3_upscale = tf.layers.conv2d(
        inputs=vgg_layer3_out, 
        filters=num_classes,
        kernel_size=1, 
        strides=(1,1),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "score_pool3")

    input = tf.add(input, pool3_upscale, name="skip_pool3")

    input = tf.layers.conv2d_transpose(
        inputs=input, 
        filters=num_classes,
        kernel_size=16, 
        strides=(8,8),
        padding="SAME",
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        name = "decoder_layer3")

    return input

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, l2_const):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # logits and labels must have the same shape, e.g. [batch_size, num_classes] and the same dtype (either float16, float32, or float64).
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    # calculate the total l2 regularization
    l2_loss = cross_entropy_loss + l2_const * tf.losses.get_regularization_loss()
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    return (logits, train_op, l2_loss)

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_value, learning_rate_value):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Starting Training")
    
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            
            _, loss = sess.run(
                       [train_op,cross_entropy_loss],
                       feed_dict={
                           input_image: image,
                           correct_label: label,
                           keep_prob: keep_prob_value,
                           learning_rate: learning_rate_value
                       })
        print("  Epoch {} - Loss = {:.3f}".format(epoch + 1, loss))

tests.test_train_nn(train_nn)

def hparam_to_string(keep_prob_value, learning_rate_value, l2_regularization_value):
    return "kp_%.0E,lr_%.0E,l2_%.0E" % (keep_prob_value, learning_rate_value, l2_regularization_value)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learningRate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--keepProb',
                        default=0.5,
                        type=float)
    parser.add_argument('--l2_regularization',
                        default=1e-3,
                        type=float)
    parser.add_argument('--epochs',
                        default=48,
                        type=int)
    parser.add_argument('--batch_size',
                        default=8,
                        type=int)
    args = parser.parse_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate_value = args.learningRate
    keep_prob_value = args.keepProb
    l2_regularization_value = args.l2_regularization
    print('epochs:              ', epochs)
    print('batch_size:          ', batch_size)
    print('learningRate:        ', keep_prob_value)
    print('keep_prob:           ', keep_prob_value)
    print('l2_regularization:   ', l2_regularization_value)
    hparam_str = hparam_to_string(keep_prob_value, learning_rate_value, l2_regularization_value)

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(dtype = tf.float32)

        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_regularization_value)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes, l2_regularization_value)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(
            sess=sess, 
            epochs=epochs,
            batch_size=batch_size,
            get_batches_fn=get_batches_fn, 
            train_op=train_op, 
            cross_entropy_loss=cross_entropy_loss, 
            input_image=image_input,
            correct_label=correct_label, 
            keep_prob=keep_prob,
            learning_rate=learning_rate,
            keep_prob_value=keep_prob_value,
            learning_rate_value=learning_rate_value)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(hparam_str, runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

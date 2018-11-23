import random

from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from frcnn.vgg import nn_base, rpn
from frcnn.config import Config
from frcnn.simple_parser import get_data
import frcnn.frcnn_losses as loss_fn
import pickle

def train():
    cfg = Config()
    all_data, classes_count, class_mapping = get_data(file_path=cfg.simple_label_filepath)
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    cfg.classes_count = classes_count
    cfg.class_mapping = class_mapping

    # Save config file for referencing while testing
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))

    # Displaying statistics of the data
    print("Total number of training images: {}".format(len(all_data)))
    print("Number of training data per class: {}".format(classes_count))
    print("Class Mapping: {}".format(class_mapping))

    # Randomly make train and test data
    random.shuffle(all_data)

    train_imgs = [s for s in all_data if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_data if s['imageset'] == 'test']

    print('Number of  train samples: {}'.format(len(train_imgs)))
    print('Number of validation samples: {}'.format(len(val_imgs)))

    input_shape_img = (None, None, 3) # for tensorflow backend

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network
    base_network_activation = nn_base(img_input, trainable=True)

    # define rpn based on the base network
    num_anchors = len(cfg.anchor_box_ratios) * len(cfg.anchor_box_scales)
    rpn_network = rpn(base_network_activation, num_anchors)

    model_rpn = Model(img_input, rpn[:2])
    rpn_optimizer = Adam(lr=1e-5)
    model_rpn.compile(rpn_optimizer, loss=[loss_fn.rpn_loss_classification(num_anchors),
                                           loss_fn.rpn_loss_regression(num_anchors)])

    epoch_len = 1000
    num_epochs = cfg.num_epochs




if __name__ == '__main__':
    train()
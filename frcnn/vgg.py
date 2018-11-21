from keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D
import keras.backend as K

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length / 16
    return get_output_length(width), get_output_length(height)

def nn_base(input_tensor=None, trainable=False):
    if K.image_dim_ordering() == 'tf':
        input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    # Block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(x)

    # Block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(x)

    # Block3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)

    # Block4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)

    # Block5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)

    return x


def rpn(base_activation, num_anchors):
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='normal', name='rpn_conv1')(base_activation)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_cls')(x)
    
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_regr')(x)

    return [x_class, x_regr, base_activation]



if __name__ == '__main__':
    nn_base()

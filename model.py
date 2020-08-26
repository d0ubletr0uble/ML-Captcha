from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.layers.experimental.preprocessing import Rescaling
from keras import Model


def get_model():
    inputs = Input(shape=(60, 200, 1))

    outputs = Rescaling(1. / 255)(inputs)
    outputs = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(outputs)
    outputs = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(outputs)
    outputs = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(outputs)
    outputs = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Flatten()(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = [
        Dense(10, name='digit1', activation='softmax')(outputs),
        Dense(10, name='digit2', activation='softmax')(outputs),
        Dense(10, name='digit3', activation='softmax')(outputs),
        Dense(10, name='digit4', activation='softmax')(outputs),
        Dense(11, name='digit5', activation='softmax')(outputs),
    ]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile()
    return model

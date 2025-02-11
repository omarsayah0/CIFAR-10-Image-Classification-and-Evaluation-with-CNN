from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train, x_test, y_test)

def set_model():
    model = keras.Sequential([
        layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(256, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(10, activation = 'softmax')
    ])
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return (model)

def main():
    x_train, y_train, x_test, y_test = load_data()
    model = set_model()
    step_per_epoch = x_train.shape[0] // 64
    data_gen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
    )
    model.fit(
        data_gen.flow(x_train, y_train, batch_size = 64),
        steps_per_epoch = step_per_epoch,
        epochs = 50,
        validation_data = (x_test, y_test)
    )
    model.save("cifar_cnn.keras")

if __name__ == '__main__':
    main()
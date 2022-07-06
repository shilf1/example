import tensorflow as tf
import datetime
import numpy as np

mnist_raw = np.load('mnist.npz')

x_train, y_train, x_test, y_test = mnist_raw['x_train'], mnist_raw['y_train'], mnist_raw['x_test'], mnist_raw['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=2000, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])


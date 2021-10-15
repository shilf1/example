
import tensorflow as tf
import os

os.system('date')
print('\n\n')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('\n\n\n\n')

os.system('ls /usr/lib/x86_64-linux-gnu/libcuda.so.1 -la')

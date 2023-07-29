import tensorflow as tf
import pandas as pd
import pickle
import sys



modelnames = 'machined_Model_30x45_2Layers_16_128_BATCH_1024_REG_l2_DROPOUT_0'
modeloutputnames = 'resize_model'

print(modelnames)
print(modeloutputnames)


# Convert the model
tf_model = tf.keras.models.load_model(f'saved_model/{modelnames}')
print(tf_model.summary())

p = pickle.dumps(tf_model)
size = sys.getsizeof(p)
print("Model size: %0.0f Bytes" % sys.getsizeof(p))



converter = tf.lite.TFLiteConverter.from_saved_model(f'saved_model/{modelnames}') # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
p = pickle.dumps(tflite_model)
size = sys.getsizeof(p)
print("Model size: %0.0f Bytes" % sys.getsizeof(p))


# Save the model.
with open(f'ported_models/{modeloutputnames}.tflite', 'wb') as f:
  f.write(tflite_model)





import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

vocab_size = 32000 #sesuai library bert-base-indonesian-522M
oov = "<OOV>"
padding = "post"
truncate = "post"
embedding_dim = 16
max_length = 25

num_classes = 5

from transformers import BertTokenizer

model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name,
    model_max_length=max_length,  # Maximum length of sequences after tokenization
    padding=padding,  # Padding strategy
    truncation=truncate,  # Truncation strategy
    max_length=max_length,  # Maximum length of sequences after padding or truncation
)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Model_Mihu_970709.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (replace with your actual input data)
masukan = "pipa saya bocor"

input_data = [tokenizer(masukan)['input_ids']]
input_data = np.array(pad_sequences(input_data, padding=padding, maxlen=max_length, truncating=truncate), dtype=np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the predictions
print("Predictions:", output_data)
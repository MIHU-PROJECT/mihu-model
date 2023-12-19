import functions_framework
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.cloud import storage
import json

# Connect to Cloud Storage and download the model
storage_client = storage.Client()
bucket = storage_client.get_bucket('mihu_model')
blob = bucket.blob('Model_Mihu_970709.tflite')
blob.download_to_filename('/tmp/Model_Mihu_970709.tflite')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/tmp/Model_Mihu_970709.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)

    # Use the loaded tokenizer for tokenization
    input_text = request_json['sentences']
    input_text = [tokenizer(input_text)['input_ids']]
    #pad token
    tokens = np.array(pad_sequences(input_text, padding=padding, maxlen=max_length, truncating=truncate), dtype=np.float32)

    print('padded:', tokens)

    #predict
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], tokens)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print('prediction:', output_data)

    return json.dumps(output_data[0].tolist())
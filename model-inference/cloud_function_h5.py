import functions_framework
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.cloud import storage
import json

# Connect to Cloud Storage and download the model
storage_client = storage.Client()
bucket = storage_client.get_bucket('mihu_model')
blob = bucket.blob('Model_Mihu_9704.h5')
blob.download_to_filename('/tmp/Model_Mihu_9704.h5')

loaded_model = tf.keras.models.load_model("/tmp/Model_Mihu_9704.h5")

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
    tokens = pad_sequences(input_text, padding=padding, maxlen=max_length, truncating=truncate)

    print('padded:', tokens)

    #predict
    prediction = loaded_model.predict(tokens)

    print('prediction:', prediction)

    return json.dumps(rediction[0])
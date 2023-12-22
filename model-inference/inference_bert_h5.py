import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

loaded_model = tf.keras.models.load_model("drive/MyDrive/Model_Mihu_9704.h5")

vocab_size = 32000 #sesuai library bert-base-indonesian-522M
oov = "<OOV>"
padding = "post"
truncate = "post"
embedding_dim = 16
max_length = 25

num_classes = 5

#set Tokenizer
import transformers
from transformers import BertTokenizer

model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name,
    model_max_length=max_length,  # Maximum length of sequences after tokenization
    padding=padding,  # Padding strategy
    truncation=truncate,  # Truncation strategy
    max_length=max_length,  # Maximum length of sequences after padding or truncation
)

# Use the loaded tokenizer for tokenization
input_text = "tv saya rusak"
input_text = [tokenizer(input_text)['input_ids']]

print(input_text)

#pad token
tokens = pad_sequences(input_text, padding=padding, maxlen=max_length, truncating=truncate)

#predict
prediction = loaded_model.predict(tokens)
print(prediction)
# mihu-model

all files associated with MIHU machine learning model development

- ipynb files
- initial saved .h5 model
- .tflite model
- inference script
- script used for cloud functions

### To Deploy the Model
download `model-inference/cloud_function_tflite.py` file

download `model-development/Model_Mihu_970709.tflite` file

put the `Model_Mihu_970709.tflite` to cloud storage

add cloud function and change the `cloud_function_tflite.py` according to the cloud storage bucket name

deploy

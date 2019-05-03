import logging

from predict_client.prod_client import ProdClient


#### export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'



#From Local Package-----------------------------
from run_multilabels_classifier import MultiLabelTextProcessor 
from run_multilabels_classifier import convert_single_example 
from run_multilabels_classifier import create_int_feature
import tokenization

import numpy as np
import collections
#-------------------------------------------------------

import tensorflow as tf





logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = 'localhost:8500'
model_name = 'bert'
model_version = 1




#-------------------------------------------------------------------#
#----------------INPUT STRING----------------------------
user_text = 'i hate all twitter users'
request_id = np.zeros((128), dtype=int).tolist()
content = {'user_text':user_text,}
label_list =[0,0,0,0,0,0]

VOCAB_FILE = 'vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

processor = MultiLabelTextProcessor()

inputExample = processor.serving_create_example([request_id, content['user_text']], 'test')
feature = convert_single_example(0, inputExample, label_list, 128, tokenizer)


features = collections.OrderedDict()
features["input_ids"] = create_int_feature(feature.input_ids)
features["input_mask"] = create_int_feature(feature.input_mask)
features["segment_ids"] = create_int_feature(feature.segment_ids)
features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
if isinstance(feature.label_id, list):
    label_ids = feature.label_id
else:
    label_ids = [feature.label_id]
features["label_ids"] = create_int_feature(label_ids)

tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#----------------------------------------------------------------------------#
tf_example = tf_example.SerializeToString()

print('test')




client = ProdClient(host, model_name, model_version)



prediction = client.predict(tf_example, request_timeout=10)
logger.info('Prediction: {}'.format(prediction))

import tensorflow as tf
from transformers import TFAutoModel, TFBertTokenizer, AutoTokenizer

# see here: https://www.philschmid.de/tensorflow-sentence-transformers
# allows us to avoid installing pytorch next to TF since
# overcooked used TF but sbert.net uses torch
# default trainable as False (freezes layers) 
# and training as false for call (keeps batch norm in inference mode, see here https://www.tensorflow.org/guide/keras/transfer_learning)
class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_name_or_path, trainable=False, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model
        # print("eager: ", tf.executing_eagerly())
        # print("functions eager: ", tf.config.functions_run_eagerly())
        # tf.config.run_functions_eagerly(True)
        # print("functions eager: ", tf.config.functions_run_eagerly())
        self.model = TFAutoModel.from_pretrained(model_name_or_path, from_pt=False, **kwargs)
        self.model.trainable = trainable
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, inputs):
        # tokenize
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='tf')
        return self(encoded_inputs)

    def call(self, inputs, normalize=True, training=False):
        # runs model on inputs
        model_output = self.model(inputs, training=training)
        # Perform pooling. In this case, mean pooling.
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        # normalizes the embeddings if wanted
        if normalize:
          embeddings = self.normalize(embeddings)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)

    def normalize(self, embeddings):
      embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
      return embeddings
    
    def set_trainable(self, trainable):
        self.model.trainable = trainable
       
    @classmethod
    def from_model_name(cls, model_name='sentence-transformers/all-MiniLM-L6-v2'):
       return cls(model_name)

class E2ESentenceTransformer(tf.keras.Model):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()
        # loads the in-graph tokenizer
        self.tokenizer = TFBertTokenizer.from_pretrained(model_name_or_path, **kwargs)
        # loads our TFSentenceTransformer
        self.model = TFSentenceTransformer(model_name_or_path, **kwargs)

    def call(self, inputs):
        # runs tokenization and creates embedding
        tokenized = self.tokenizer(inputs)
        return self.model(tokenized)
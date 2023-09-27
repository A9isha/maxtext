import tensorflow_datasets as tfds; 

tfds.load('lm1b', split='test', shuffle_files=True, download=True)
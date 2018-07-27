# Bit-RNN

Source code for paper: [Effective Quantization Methods for Recurrent Neural Networks](https://arxiv.org/abs/1611.10176).

The implementation of PTB language model is modified from examples in [tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/ptb).

## Requirments

Currently tested and run on [TensorFlow](https://www.tensorflow.org) 1.8 and Python 3.6. View other branches for legacy support.
You may download the data from [http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

## Run
```
python train.py --config=config.gru --data_path=YOUR_DATA_PATH
```

Currently default is 2-bit weights and activations. You may edit the config file in config folder to change configuration.

## Support
Submit issue for problem relate to the code itself. Send email to the author for general question about the paper.

## Citation
Please cite follow if you use our code in your research:
```
@article{DBLP:journals/corr/HeWZWYZZ16,
  author    = {Qinyao He and
               He Wen and
               Shuchang Zhou and
               Yuxin Wu and
               Cong Yao and
               Xinyu Zhou and
               Yuheng Zou},
  title     = {Effective Quantization Methods for Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1611.10176},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.10176},
  timestamp = {Thu, 01 Dec 2016 19:32:08 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/HeWZWYZZ16},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

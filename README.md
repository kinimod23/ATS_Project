# Automatic Text Simplification Project

* Bytenet https://arxiv.org/abs/1610.10099, https://github.com/paarthneekhara/byteNet-tensorflow
* Wikipedia Korpus http://www.cs.pomona.edu/~dkauchak/simplification/
* Deep Voice https://arxiv.org/abs/1710.07654
* Other papers: http://www.thespermwhale.com/jaseweston/papers/unified_nlp.pdf \\ https://arxiv.org/pdf/1404.2188.pdf
* ABCNN: https://arxiv.org/pdf/1512.05193.pdf
* Recursive Autoencoder for Paraphrase Detection https://papers.nips.cc/paper/4204-dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection.pdf
* Convolutional NN (translation) https://github.com/tobyyouup/conv_seq2seq
----------------------------------------------------------------------------------------
## ToDo
* statistics about data  ----------------> Dominik bis 23.08.
  * how long are sentences (how many are lost if we cut at 50 words)
  
  cut at 40:
  85.94% are remaining
  
  cut at 50:
  94.18% are remaining
  
  
  * how many sentences have 30%, 40%, 50% of words unknown to word2vec
  (continued with cut at 40)
  
  10% unknown to word2vec:
  0.47% remaining
  
  20% unknown to word2vec:
  9.57% remaining
  
  30% unknown to word2vec:
  41.20% remaining
  
  40% unknown to word2vec:
  68.22% remaining
  
  50% unknown to word2vec:
  79.77% remaining

  
  * how does this change with fasttext
  
  1% unknown to fasttext:
  46.67% remaining
  
  5% unknown to fasttext:
  54.69% remaining
  
  10% unknown to fasttext:
  70.68% remaining
  
  20% unknown to fasttext:
  82.86% remaining
  
  50% unknown to fasttext:
  85.87% remaining

  
  
* change test.py            ----------------> Simon bis 21.08.
  * ( creates output now )
* now using fasttext...
 * train on different data ( Wikipedia Data )
* think about other ideas for loss functions
  * (currently L2-Norm)
  * and also Cosine Similarity for End2End
  * maybe better another matrix similarity measure
* Get deconvolution with pretrained model to work ------------------> Simon/Tim 26.08.
  * This is really a pain in the ass, i don't know how to do it

----------------------------------------------------------------------------------------
## What we have done
* Implemented model with attention and all that
    * tried a lot but had exploding gradients
* removed attention, since it doesn't make sense anyway
    * because we can't feed complex and simple for sentences other than our training set
      but that is the goal right?
* restructured code to allow for siamese net as encoder ( concolutional ) and deconvolution as decoder. also to use both together as End2End
    * deconvolution hard to implement because of lack of knowledge regarding tensorflows options for that
    * implemented an image deconvolution with tf.image.resize (not real deconvolution i think)
* used Word2Vec embeddings trained on 100billion words (google news)
    * there were a lot of unknown words hence we switched to FAIRs fasttext trained on english wikipedia
* tried End2End in various forms, for thousends of epochs but didn't get good results
    * changed cost function
* tried to implement the loading and restoring of models so that we can train the encoder with cosine similarity as target and later on use this pretrained thing to create inputs for the decoder
    * took me a lot of hours to get this to work. i rebuilt the model various times and now we have a splitted model. one for encoder one for decoder

----------------------------------------------------------------------------------------
#### feasible NN implementations we could use

*CNN implementation:*

* _Attention-based CNN for modeling sentence pairs_
-> [github](https://github.com/galsang/ABCNN)

* Generalization of CNNs by using graph signal processing applied on any graph structure. Definition of convolutional filters on graphs.
-> [github](https://github.com/mdeff/cnn_graph)

*ByteNet implementation:*

* Tensor2Tensor library of deep learning models with bytenet partly maintained by google -> [github](https://github.com/tensorflow/tensor2tensor)

* bytenet without target network + training framework missing -> [github](https://github.com/NickShahML/bytenet_tensorflow)

* bytenet trained on eng-fr corpus. Relies on TF v1 -> [github](https://github.com/buriburisuri/ByteNet)

* generation trained on shakespeare and translation on ger-en -> [github](https://github.com/paarthneekhara/byteNet-tensorflow)

*Recursive Neural Nets:*
* Recursive Neural Networks with tree structure in Tensorflow -> [github](https://github.com/erickrf/treernn)

#### Project Organisation

#### A short memorable project title.
Convolutional Syntax Tree Translation

#### What is the problem you want to address? How do you determine whether you have solved it?
Taking syntax trees of complex sentences, we want to "translate" them into one or moresyntax trees of simpler sentences while keeping the meaning.
Solved?

#### How is this going to translate into a machine learning problem?
Neural Net consisting of encoder and decoder stage.
Two parallel encoder sharing the same weights, one with complex syntax trees as input, the other with the simplified version.
The outputs of the encoder are compared with an appropriate metric and a discriminator.

##### Why do you think deep learning will help to solve it? Is there any related literature that backs your claim?
Approaches in Machine Translation yield state-of-the-art results. Interpreting our task as translation problem, we can use existing ideas and extend them to our needs.
Stajner et al. (2017)

##### Which data are you planning to use?
( Wikipedia simple and complex corpus )
The above but improved by Hwang et al. (2015)

##### Which specific steps (e.g. milestones) are you going to take towards solving the problem? What's the schedule?
- determine if the hypothesis of comparability of complex and simplified syntax trees is correct (group effort)
  - create Syntax trees out of the data (Henny)
  - evaluate some example sentences manually (group effort)
- Find a good representation of syntax trees in the literature (group effort)
- Modify an existing ByteNet implementation to take the new input and visualize it in Tensorboard (Simon)
- Organise the data such that an encoder can be trained properly
- Train an encoder, find out if it converges
- Train a decoder with simplified syntax trees as gold standard

##### How will the work be distributed among the team members?

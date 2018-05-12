# Automatic Text Simplification Project

* Bytenet https://arxiv.org/abs/1610.10099, https://github.com/paarthneekhara/byteNet-tensorflow
* Wikipedia Korpus http://www.cs.pomona.edu/~dkauchak/simplification/
* Deep Voice https://arxiv.org/abs/1710.07654
* Other papers: http://www.thespermwhale.com/jaseweston/papers/unified_nlp.pdf \\ https://arxiv.org/pdf/1404.2188.pdf


## feasible NN implementations we could use
Generalization of CNNs by using graph signal processing applied on any graph structure. Definition of convolutional filters on graphs. 
[GITHUB: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://github.com/mdeff/cnn_graph)

ByteNet implementation:
[Tensor2Tensor library of deep learning models (with bytenet) partly maintained by google](https://github.com/tensorflow/tensor2tensor) 
[bytenet without target network + training framework missing](https://github.com/NickShahML/bytenet_tensorflow)
[bytenet trained on eng-fr corpus. Relies on TF v1](https://github.com/buriburisuri/ByteNet)
[generation trained on shakespeare and translation on ger-en](https://github.com/paarthneekhara/byteNet-tensorflow)

[Recurrent Neural Networks with tree structure in Tensorflow](https://github.com/erickrf/treernn)

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

# Automatic Text Simplification Project

* Bytenet https://arxiv.org/abs/1610.10099, https://github.com/paarthneekhara/byteNet-tensorflow
* Wikipedia Korpus http://www.cs.pomona.edu/~dkauchak/simplification/
* Deep Voice https://arxiv.org/abs/1710.07654
* Other papers: http://www.thespermwhale.com/jaseweston/papers/unified_nlp.pdf \\ https://arxiv.org/pdf/1404.2188.pdf

# Milestones and shedule
- determine if the hypothesis of comparibatility of complex and simplified syntax trees is correct
  - create Syntax trees out of the data
  - evaluate some example sentences manually
- Find a good representation of syntax trees in the literature
- Modify an existing ByteNet implementation to take the new input and visualize in Tensorboard
- Organise the data such that an encoder can be trained properly
- Train an encoder, find out if it converges
- Train a decoder with simplified syntax trees as gold standard

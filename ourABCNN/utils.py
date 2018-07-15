
def build_path(prefix, model, num_layers, model_type, word2vec):
    return prefix + model + "-" + model_type + '-' + str(num_layers) + '-' + word2vec + '-Layers'

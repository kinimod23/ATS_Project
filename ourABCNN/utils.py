
def build_path(prefix, data, model, num_layers, model_type, word2vec):
    return prefix + model + "-" + data + '-' + model_type + '-' + str(num_layers) + '-Layers' + '-' + word2vec

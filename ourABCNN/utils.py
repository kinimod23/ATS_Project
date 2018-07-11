
def build_path(prefix, model, num_layers, model_type):
    return prefix + model + "-" + model_type + '-' + str(num_layers) + '-Layers'

def feature_selection(regression_features_raw, regression_labels, frac):
    ff = np.zeros((regression_features_raw.shape[1], 19))
    for p in range(19):
        ff[:,p], _ = f_regression(regression_features_raw, regression_labels[:,p])
    ff = np.nanmean(ff, axis = 1)

    features_to_keep = np.argsort(ff)[-int(ff.shape[0]*frac):]

    regression_features = np.zeros((regression_features_raw.shape[0], len(features_to_keep)))

    for i, f in enumerate(features_to_keep):
        regression_features[:,i] = regression_features_raw[:,f]
    return regression_features, features_to_keep

def feature_selection_test_set(regression_features_raw, features_to_keep):
    regression_features = np.zeros((regression_features_raw.shape[0], len(features_to_keep)))

    for i, f in enumerate(features_to_keep):
        regression_features[:,i] = regression_features_raw[:,f]
    return regression_features

def add_task_specific_metafeatures(regression_features_raw, task_specific_features, tasks_list, sample_size):
    regression_features_raw[regression_features_raw==-np.inf] = 0
    regression_features_raw = np.concatenate([regression_features_raw, np.zeros((len(tasks_list)*sample_size, 5))],axis=1)
    for task_id, task in enumerate(tasks_list):
        for nr in range(sample_size):
            regression_features_raw[task_id*sample_size+nr,-5:] = task_specific_features[task_id,:]
    return regression_features_raw
def joint_normalize_metafeatures(metafeatures1, metafeatures2):
    """
    Normalization of two sets of metafeatures. Input is two set of metafeatures, output is the same set but normalized. Normalized metafeatures have mean 0 and standard deviation 1
    """
    normalized_metafeatures1 = np.zeros_like(metafeatures1)
    normalized_metafeatures2 = np.zeros_like(metafeatures2)
    metafeatures_joint = np.concatenate([metafeatures1, metafeatures2], axis = 0)
    for c in range(metafeatures_joint.shape[1]):
        std = np.std(metafeatures_joint[:,c])
        if std == 0:
            std = 1
        normalized_metafeatures1[:,c] = (metafeatures1[:,c]-np.mean(metafeatures_joint[:,c]))/std
        normalized_metafeatures2[:,c] = (metafeatures2[:,c]-np.mean(metafeatures_joint[:,c]))/std
    return normalized_metafeatures1, normalized_metafeatures2

def normalize_metafeatures(metafeatures):
    """
    Normalization of metafeatures. Normalized metafeatures have mean 0 and standard deviation 1
    """
    normalized_metafeatures = np.zeros_like(metafeatures)
    for c in range(metafeatures.shape[1]):
        std = np.std(metafeatures[:,c])
        if std == 0:
            std = 1
        normalized_metafeatures[:,c] = (metafeatures[:,c]-np.mean(metafeatures[:,c]))/std

    return normalized_metafeatures

def simple_single_model(nr_of_features = 1):
    model = Sequential()
    model.add(Dense(50, input_dim=nr_of_features, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))#,activity_regularizer=l1(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
    return model

task_specific_features = np.array([ [0, 0, 0.39, 1.0, 1],
                                    [1, 0, 0.10, 1.0, 0],
                                    [1, 1, 1.00, 0.7, 0],
                                    [1, 0, 1.00, 1.0, 0],
                                    [1, 0, 0.51, 1.0, 0],
                                    [0, 1, 0.06, 0.1, 1],
                                    [1, 1, 0.10, 0.4, 0],
                                    [0, 1, 0.13, 0.1, 1],
                                    [1, 1, 0.21, 0.4, 0],
                                    [1, 1, 0.07, 0.4, 0],
                                    [1, 1, 1.00 ,0.7, 0],
                                    [1, 1, 1.00 ,0.7, 0],
                                    [1, 0, 1.00 ,1.0, 1]])

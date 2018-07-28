from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras import backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

img_width, img_height = 224, 224

top_model_weights_path = 'saved_weights/model.h5'

train_data_dir = 'chest_xray/train'
validation_data_dir = 'chest_xray/val'
test_data_dir = 'chest_xray/test'

train_feature_file = 'saved_features/train_features.npy'
validation_feature_file = 'saved_features/validation_features.npy'
test_feature_file = 'saved_features/test_features.npy'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16

model = VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)


def sensitivity(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = np.sum(np.round(np.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + backend.epsilon())


def get_bottleneck_features(dir, feature_file):

    generator = datagen.flow_from_directory(
        dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_samples = len(generator.filenames)

    predict_size = int(math.ceil(nb_samples / batch_size))

    bottleneck_features_train = model.predict_generator(generator, predict_size)

    np.save(feature_file, bottleneck_features_train)


def get_categorical_labels(dir, feature_file):
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(generator_top.class_indices)

    # load the bottleneck features saved earlier
    data = np.load(feature_file)

    # get the class lebels for the training data, in the original order
    labels = generator_top.classes

    # convert the training labels to categorical vectors
    labels = to_categorical(labels, num_classes=num_classes)
    return data, labels, num_classes


def train_top_model(model_weights_path, train_feature_path, val_feature_path, train_dir, val_dir):
    train_data, train_labels, num_classes = get_categorical_labels(train_dir, train_feature_path)
    validation_data, validation_labels, num_classes = get_categorical_labels(val_dir, val_feature_path)

    vgg_model = Sequential()
    vgg_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    vgg_model.add(Dense(num_classes, activation='softmax'))
    vgg_model.summary()

    vgg_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=model_weights_path, verbose=1, save_best_only=True)

    history = vgg_model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels), callbacks=[checkpointer])

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = vgg_model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    plot_history(history)


def plot_history(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def vgg16_predict_test_set(test_dir, weights_path, test_features, train_dir, train_features):
    train_data, train_labels, num_classes = get_categorical_labels(train_dir, train_features)
    test_data, test_labels, num_classes = get_categorical_labels(test_dir, test_features)

    vgg_model = Sequential()
    vgg_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    vgg_model.add(Dense(num_classes, activation='softmax'))
    vgg_model.load_weights(weights_path)
    preds = vgg_model.predict(test_data, batch_size=batch_size)
    return preds, test_labels


def single_class_accuracy(y_true, y_pred, class_id):
    class_id_true = np.argmax(y_true, axis=-1)
    class_id_preds = np.argmax(y_pred, axis=-1)
    accuracy_mask = np.cast(np.equal(class_id_preds, class_id), 'int32')
    class_acc_tensor = np.cast(np.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = np.sum(class_acc_tensor) / np.maximum(np.sum(accuracy_mask), 1)
    return class_acc


def get_normal_vs_pneumonia_metrics(preds, targets):
    class_id_preds = np.argmax(preds, axis=-1)
    class_id_true = np.argmax(targets, axis=-1)

    pred_positive = class_id_preds > 0
    true_positive = class_id_true > 0
    get_evaluation_metrics(true_positive.astype(int), pred_positive.astype(int))


def get_pneumonia_vs_others(preds, targets, class_id):
    class_id_preds = np.argmax(preds, axis=-1)
    class_id_true = np.argmax(targets, axis=-1)

    pred_positive = class_id_preds == class_id
    true_positive = class_id_true == class_id
    get_evaluation_metrics(true_positive.astype(int), pred_positive.astype(int))

def get_evaluation_metrics(y_true, y_pred):
    specificity_score = specificity(y_true, y_pred)
    sensitivity_score = sensitivity(y_true, y_pred)
    sum_correct = np.sum(np.array(y_true == y_pred))
    total_preds = len(y_pred)
    accuracy = sum_correct / total_preds

    print('accuracy {0:.4f}  specificity {1:.4f} sensitivity {2:.4f}'.format(accuracy, specificity_score, sensitivity_score))


def extract_features():
    get_bottleneck_features(train_data_dir,  train_feature_file)
    get_bottleneck_features(validation_data_dir, validation_feature_file)
    get_bottleneck_features(test_data_dir, test_feature_file)




#print('extract features')
#extract_features()

# print('train top model')
# train_top_model(model_weights_path=top_model_weights_path,
#                 train_feature_path=train_feature_file,
#                 val_feature_path=validation_feature_file,
#                 train_dir=train_data_dir,
#                 val_dir=validation_data_dir)
#

predictions, target_labels = vgg16_predict_test_set(test_dir = test_data_dir,
                       weights_path=top_model_weights_path,
                       test_features=test_feature_file,
                       train_dir=train_data_dir,
                       train_features=train_feature_file)

print('Normal vs pneumonia metrics ...')
get_normal_vs_pneumonia_metrics(predictions, target_labels)

print('Bacterial vs others metrics ...')
get_pneumonia_vs_others(predictions, target_labels, 1)

print('Viral vs others metrics ...')
get_pneumonia_vs_others(predictions, target_labels, 2)


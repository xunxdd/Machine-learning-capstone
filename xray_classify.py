from numpy.random import seed
seed(1)
import os
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras import backend
from keras.preprocessing import image
import matplotlib
import matplotlib.image as mpimg
from keras.callbacks import LearningRateScheduler
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import util
from time import gmtime, strftime
from keras.models import load_model
from sklearn.metrics import confusion_matrix

model_names = ['InceptionResNetV2']  # 'ResNet50', 'Xception', 'InceptionV3'

mode_file_processing = 'file_processing'
mode_feature_extraction = 'feature_extraction'
mode_train_fit_predict = 'train_fit_predict'
mode_train_validate = 'train_validate'
mode_fine_tune = 'fine_tune'
mode_predict = 'predict'

img_width, img_height = 224, 224
size_dir_path = '224_224'
train_data_dir = 'chest_xray/train'
validation_data_dir = 'chest_xray/val'
test_data_dir = 'chest_xray/test'
learning_rate = 0.001
dropout_rate = 0.2
# with image augmentation image size 224 by 224
train_feature_file = 'saved_features/{0}/{1}/train_features.npy'
validation_feature_file = 'saved_features/{0}/{1}/validation_features.npy'
test_feature_file = 'saved_features/{0}/{1}/test_features.npy'
saved_model_weights_path = 'saved_weights/{0}/{1}/model.h5'
saved_history_path = 'history/{0}/{1}/plot_{2}.png'

# number of epochs to train top model
epochs = 200
# batch size used by flow_from_directory and predict_generator
batch_size = 16

datagen = ImageDataGenerator(rescale=1. / 255)


def lr_decay_callback(lr_init, lr_decay):
    def step_decay(epoch):
        print(lr_init, lr_init * (lr_decay ** (epoch + 1)))
        return lr_init * (lr_decay ** (epoch + 1))
    return LearningRateScheduler(step_decay)


def check_file_folder(file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)


def pretrained_model(model_name):
    return {
        'VGG16': VGG16(include_top=False, weights='imagenet'),
        'Xception': Xception(include_top=False, weights='imagenet'),
        'InceptionResNetV2': InceptionResNetV2(weights='imagenet', include_top=False),
        'InceptionV3': InceptionV3(weights='imagenet', include_top=False),
        'ResNet50': ResNet50(weights='imagenet', include_top=False),
    }[model_name]


def sensitivity(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = np.sum(np.round(np.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = np.sum(np.round(np.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + backend.epsilon())


def get_bottleneck_features(dir, feature_file, pretrained_model):
    check_file_folder(feature_file)
    generator = datagen.flow_from_directory(
        dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_samples = len(generator.filenames)

    predict_size = int(math.ceil(nb_samples / batch_size))

    features_train = pretrained_model.predict_generator(generator, predict_size)

    np.save(feature_file, features_train)
    print('feature extracted and saved in ', feature_file)


def get_categorical_labels(dir, feature_file):
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(generator_top.class_indices)
    data = np.load(feature_file)

    labels = generator_top.classes

    labels = to_categorical(labels, num_classes=num_classes)
    return data, labels, num_classes


def get_top_model(data, num_classes):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=data.shape[1:]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_top_model(model_weights_path, train_feature_path, val_feature_path, train_dir, val_dir, model_name, current_size='224_224'):
    check_file_folder(saved_model_weights_path.format(model_name, current_size))
    train_data, train_labels, num_classes = get_categorical_labels(train_dir, train_feature_path.format(model_name, current_size))
    validation_data, validation_labels, num_classes = get_categorical_labels(val_dir,
                                                                             val_feature_path.format(model_name, current_size))

    model = get_top_model(train_data, num_classes)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=30, verbose=1, mode='auto')
    lr_decay = .99
    # 1e-4, .0005
    model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    lr_decay = lr_decay_callback(learning_rate, lr_decay)
    checkpointer = ModelCheckpoint(filepath=model_weights_path, verbose=1, save_best_only=True,
                                   save_weights_only=False, )
    callbacks_list = [earlystop, checkpointer]
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels), callbacks=callbacks_list)

    model.save_weights(saved_model_weights_path.format(model_name, current_size))

    plot_history(history, model_name, current_size)


def plot_history(history, model_name, current_size='224_224'):
    check_file_folder(saved_history_path.format(model_name, current_size,  strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    plt.figure()

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('{0} model accuracy'.format(model_name))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{0} model loss'.format(model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.savefig(saved_history_path.format(model_name, current_size,  strftime("%Y-%m-%d %H:%M:%S", gmtime())))


def predict_test_set(test_dir, weights_path, test_features, train_dir, train_features):
    train_data, train_labels, num_classes = get_categorical_labels(train_dir, train_features)
    test_data, test_labels, num_classes = get_categorical_labels(test_dir, test_features)

    model = get_top_model(train_data, num_classes)
    model.load_weights(weights_path)
    preds = model.predict(test_data, batch_size=batch_size)
    return preds, test_labels


def get_normal_vs_pneumonia_metrics(preds, targets):
    class_id_preds = np.argmax(preds, axis=-1)
    class_id_true = np.argmax(targets, axis=-1)

    pred_positive = class_id_preds > 0
    true_positive = class_id_true > 0
    return get_evaluation_metrics(true_positive.astype(int), pred_positive.astype(int))


def get_pneumonia_vs_others(preds, targets, class_id):
    class_id_preds = np.argmax(preds, axis=-1)
    class_id_true = np.argmax(targets, axis=-1)

    pred_positive = class_id_preds == class_id
    true_positive = class_id_true == class_id
    return get_evaluation_metrics(true_positive.astype(int), pred_positive.astype(int))


def get_evaluation_metrics(y_true, y_pred):
    specificity_score = specificity(y_true, y_pred)
    sensitivity_score = sensitivity(y_true, y_pred)
    sum_correct = np.sum(np.array(y_true == y_pred))
    total_preds = len(y_pred)
    accuracy = sum_correct / total_preds
    f_score = 2 * (specificity_score * sensitivity_score) / (specificity_score + sensitivity_score)
    return accuracy, specificity_score, sensitivity_score, f_score


def extract_features(model_name, current_size = '224_224'):
    pretrain_model = pretrained_model(model_name)
    get_bottleneck_features(train_data_dir, train_feature_file.format(model_name, current_size), pretrain_model)
    get_bottleneck_features(validation_data_dir, validation_feature_file.format(model_name, current_size), pretrain_model)
    get_bottleneck_features(test_data_dir, test_feature_file.format(model_name, current_size), pretrain_model)


def predict_evaluate_on_testset(current_model_name, current_size = '224_224'):
    predictions, target_labels = predict_test_set(test_dir=test_data_dir,
                                                  weights_path=saved_model_weights_path.format(current_model_name, current_size),
                                                  test_features=test_feature_file.format(current_model_name, current_size),
                                                  train_dir=train_data_dir,
                                                  train_features=train_feature_file.format(current_model_name, current_size))

    results = []
    results.append(
        '\n Model Name ' + current_model_name + ' img size ' + str(img_height) + ' pooling Max Pooling Average With adam '
        + str(learning_rate)
        + ' and '+ str(dropout_rate) + 'dropout at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    results.append(', Accuracy, Specificity, Sensitivity, F1-Score')

    accuracy, specificity_score, sensitivity_score, fscore = get_normal_vs_pneumonia_metrics(predictions, target_labels)
    results.append('Normal vs pneumonia, {0:.4f}, {1:.4f} , {2:.4f}, {3:.4f}'.format(accuracy, specificity_score,
                                                                                     sensitivity_score, fscore))

    accuracy, specificity_score, sensitivity_score, fscore = get_pneumonia_vs_others(predictions, target_labels, 1)
    results.append('Bacterial vs others metrics, {0:.4f},  {1:.4f} , {2:.4f}, {3:.4f}'.format(accuracy,
                                                                                              specificity_score,
                                                                                              sensitivity_score,
                                                                                              fscore))

    accuracy, specificity_score, sensitivity_score, fscore = get_pneumonia_vs_others(predictions, target_labels, 2)
    results.append(
        'Viral vs others metrics, {0:.4f} , {1:.4f} , {2:.4f}, {3:.4f}'.format(accuracy,
                                                                                        specificity_score,
                                                                                        sensitivity_score, fscore))

    with open("Output.csv", "a") as text_file:
        for result in results:
            text_file.write(result + '\n')


def extract_InceptionResV2(tensor):
    return pretrained_model('InceptionResNetV2').predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def predict_chest_xray(model, img_path):
    bottleneck_feature = extract_InceptionResV2(path_to_tensor(img_path))
    predicted_vector = model.predict(bottleneck_feature)
    return np.max(predicted_vector), np.argmax(predicted_vector)


def predict_images(model_name, current_size):
    imgdir = 'chest-xray-sample-images'

    model = load_model(saved_model_weights_path.format(model_name, current_size))

    for img in os.listdir(imgdir):
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = imgdir + '/' + img
            print(img_path)
            # img = mpimg.imread(img_path)
            # imgplot = plt.imshow(img)
            # plt.show()
            predictName, res = predict_chest_xray(model, img_path)
            print (predictName, res)


def main():
    current_model_name = 'InceptionResNetV2'
    current_size = '224_224'
    current_mode = mode_fine_tune
    global img_width
    global img_height
    global learning_rate
    global dropout_rate

    if current_mode == mode_file_processing:
        util.move_files()

    if current_mode == mode_feature_extraction:
        for mode_name in model_names:
            current_model_name = mode_name
            print('extract features model name ', current_model_name)
            extract_features(current_model_name)

    if current_mode == mode_train_validate:
        for mode_name in model_names:
            current_model_name = mode_name
            top_model_weights_path = saved_model_weights_path.format(current_model_name, current_size)
            train_top_model(model_weights_path=top_model_weights_path,
                            train_feature_path=train_feature_file,
                            val_feature_path=validation_feature_file,
                            train_dir=train_data_dir,
                            val_dir=validation_data_dir, model_name=current_model_name)

    if current_mode == mode_train_fit_predict:
        for mode_name in model_names:
            current_model_name = mode_name
            predict_evaluate_on_testset(current_model_name)

    if current_mode == mode_fine_tune:
        learning_rate = 0.0005
        for size in [299]: #[299,320]
            if size == 299:
                img_height, img_width = 299, 299
                current_size = '299_299'
            elif size == 320:
                img_height, img_width = 320, 320
                current_size = '320_320'
            print(size, ': ',  train_feature_file, validation_feature_file, test_feature_file, saved_model_weights_path, saved_history_path)
            top_model_weights_path = saved_model_weights_path.format(current_model_name, current_size)
            train_top_model(model_weights_path=top_model_weights_path,
                            train_feature_path=train_feature_file,
                            val_feature_path=validation_feature_file,
                            train_dir=train_data_dir,
                            val_dir=validation_data_dir, model_name=current_model_name, current_size=current_size)
            predict_evaluate_on_testset(current_model_name, current_size=current_size)

    if current_mode == mode_predict:
        predict_images(current_model_name)


if __name__ == "__main__":
    main()

from glob import glob
import os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from keras.backend import cast_to_floatx
from keras.preprocessing.image import apply_affine_transform, random_shift, random_rotation, load_img

def resize_and_crop_image(input_file, output_side_length, greyscale=False):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    height_offset = (new_height - output_side_length) // 2
    width_offset = (new_width - output_side_length) // 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    assert cropped_img.shape[:2] == (output_side_length, output_side_length)
    return cropped_img

def getData(PATH, cnames):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for idx, i in enumerate(cnames):
        train_path = PATH+'/train/'+str(i)
        train_batch = os.listdir(train_path)

        for sample in train_batch:
                img_path = train_path+'/'+sample
                x = load_img(img_path).resize((32,32))
                # preprocessing if required
                x_train.append(np.array(x))
                y_train.append(idx)
 
    test_path = PATH+'/test/'
    test_batch = os.listdir(test_path)

    for idx, i in enumerate(cnames):
        test_path = PATH+'/test/'+str(i)
        test_batch = os.listdir(test_path)

        for sample in test_batch:
                img_path = test_path+'/'+sample
                x =load_img(img_path).resize((32,32))
                # preprocessing if required
                x_test.append(np.array(x))
                y_test.append(idx)
    idx = np.random.permutation(len(y_test))
    y_test = y_test[idx]
    x_test = x_test[idx]	
    # finally converting list into numpy array
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def getDataNoCrop(PATH, cnames):
    x_train = []
    y_train = []
    for idx, i in enumerate(cnames):
        train_path = PATH+'/'+str(i)
        train_batch = os.listdir(train_path)

        for sample in train_batch:
                img_path = train_path+'/'+sample
                x = load_img(img_path).resize((32,32))
                # preprocessing if required
                x_train.append(np.array(x))
                y_train.append(idx)
    idx = np.random.permutation(len(y_train))
    y_train = np.array([y_train[i] for i in idx])
    x_train = np.array([x_train[i] for i in idx]) # finally converting list into numpy array
    return (np.array(x_train), np.array(y_train))
def normalize_minus1_1(data):
    return 2*(data/255.) - 1


def get_channels_axis():
    import keras
    idf = keras.backend.image_data_format()
    if idf == 'channels_first':
        return 1
    assert idf == 'channels_last'
    return 3


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)



def load_mnist():
    (X_train, y_train), (X_test, y_test) = getData('MNIST', [0,1,2,3,4,5,6,7,8,9])
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    print([len(X_train),len(X_test)])
    return (X_train, y_train), (X_test, y_test)

def load_amazon():
    (X_train, y_train) = getDataNoCrop('data/office/amazon/images', range(31))
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = X_train[int(0.8*len(X_train)):]
    X_train = X_train[0:int(0.8*len(X_train))]
    y_test = y_train[int(0.8*len(y_train)):]
    y_train = y_train[0:int(0.8*len(y_train))]
    print([len(X_train),len(X_test)])
    return (X_train, y_train), (X_test, y_test)


def load_dslr():
    (X_train, y_train) = getDataNoCrop('data/office/dslr/images', range(31))
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = X_train[int(0.8*len(X_train)):]
    X_train = X_train[0:int(0.8*len(X_train))]
    y_test = y_train[int(0.8*len(y_train)):]
    y_train = y_train[0:int(0.8*len(y_train))]
    print([len(X_train),len(X_test)])
    return (X_train, y_train), (X_test, y_test)


def load_svhn():
    (X_train, y_train), (X_test, y_test) = getData('/home/labuser/Downloads/Preprocess-SVHN-master', [0,1,2,3,4,5,6,7,8,9])
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    print([len(X_train),len(X_test)])
    return (X_train, y_train), (X_test, y_test)


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #idx= np.random.permutation(len(X_train))
    #idx = idx[0:150]	
    #X_train = X_train[idx][:][:][:]
    #y_train = y_train[idx]

    #for i in range(np.shape(X_test)[0]):
	#X_test[i,:,:,:]  = random_shift(X_test[i,:,:,:], 0.25, 0.25, channel_axis=2, fill_mode='reflect')
	#X_test[i,:,:,:] = apply_affine_transform(np.squeeze(X_test[i,:,:,:]), tx=0, ty=0, theta=45, channel_axis=2, fill_mode='reflect')
	#X_test[i,:,:,:]  = random_rotation( X_test[i,:,:,:]  , 45, channel_axis=2, fill_mode='reflect')
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def load_cifar100(label_mode='coarse'):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def create_cats_vs_dogs_npz(cats_vs_dogs_path='./'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
        imgs_paths = glob(glob_path)
        images = [resize_and_crop_image(p, 64) for p in imgs_paths]
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('train')
    x_test, y_test = _load_from_dir('test')

    np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)


def load_cats_vs_dogs(cats_vs_dogs_path='./'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
    x_train = normalize_minus1_1(cast_to_floatx(npz_file['x_train']))
    y_train = npz_file['y_train']
    x_test = normalize_minus1_1(cast_to_floatx(npz_file['x_test']))
    y_test = npz_file['y_test']

    return (x_train, y_train), (x_test, y_test)


def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
    }

    return ind_to_name[dataset_name][index]

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import Sequence
from PIL import Image
import numpy as np
import csv

LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(34)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


class DataGenerator(Sequence):

    def __init__(self,
                 list_IDs,
                 labels,
                 batch_size=32,
                 dim=(200, 60),
                 n_channels=3,
                 n_classes=34,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = np.array(
                Image.open(f'./data/5_imitate_train_set/{ID}.jpg')) / 255.0
            # Store class
            y[i] = self.labels[ID]
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


# Create CNN Model
print("Creating CNN model...")
input = Input((60, 200, 3))
out = input
out = Conv2D(
    filters=32, kernel_size=(3, 3), padding='same', activation='relu')(
        out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(
    filters=64, kernel_size=(3, 3), padding='same', activation='relu')(
        out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(
    filters=128, kernel_size=(3, 3), padding='same', activation='relu')(
        out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.3)(out)
out = [
    Dense(34, name='digit1', activation='softmax')(out),
    Dense(34, name='digit2', activation='softmax')(out),
    Dense(34, name='digit3', activation='softmax')(out),
    Dense(34, name='digit4', activation='softmax')(out),
    Dense(34, name='digit5', activation='softmax')(out)
]
model = Model(inputs=input, outputs=out)
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

partition = dict(train=[], validation=[])
labels = dict()

print("Reading training data...")
traincsv = open(
    './data/5_imitate_train_set/captcha_train.csv', 'r', encoding='utf8')
train_labels = {
    f'train-{row[0]})': np.asarray(toonehot(row[1]))
    for row in csv.reader(traincsv)
}
partition['train'] = list(train_labels.keys())
labels.update(train_labels)

print("Reading validation data...")
traincsv = open(
    './data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding='utf8')
train_labels = {
    f'vali-{row[0]}': np.asarray(toonehot(row[1]))
    for row in csv.reader(traincsv)
}
partition['validation'] = list(train_labels.keys())
labels.update(train_labels)

training_datagen = DataGenerator(
    partition['train'],
    labels,
    batch_size=32,
    dim=(200, 60),
    n_channels=3,
    n_classes=34)
validation_datagen = DataGenerator(
    partition['validation'],
    labels,
    batch_size=32,
    dim=(200, 60),
    n_channels=3,
    n_classes=34)
# traincsv = open(
#     './data/5_imitate_train_set/captcha_train.csv', 'r', encoding='utf8')
# train_data = (
#     np.array(Image.open("./data/5_imitate_train_set/" + row[0] + ".jpg")) /
#     255.0 for row in csv.reader(traincsv))
# traincsv = open(
#     './data/5_imitate_train_set/captcha_train.csv', 'r', encoding='utf8')
# read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
# train_label = [[] for _ in range(5)]
# for arr in read_label:
#     for index in range(5):
#         train_label[index].append(arr[index])
# train_label = (arr for arr in np.asarray(train_label))
# train = ((data, label) for data, label in zip(train_data, train_label))
# print("Shape of train data:", train_data.shape)

# valicsv = open(
#     './data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding='utf8')
# vali_data = (
#     np.array(Image.open("./data/5_imitate_vali_set/" + row[0] + ".jpg")) / 255.0
#     for row in csv.reader(valicsv))
# valicsv = open(
#     './data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding='utf8')
# read_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
# vali_label = [[] for _ in range(5)]
# for arr in read_label:
#     for index in range(5):
#         vali_label[index].append(arr[index])
# vali_label = (arr for arr in np.asarray(vali_label))
# vali = ((data, label) for data, label in zip(vali_data, vali_label))
# print("Shape of validation data:", vali_data.shape)

filepath = "./data/model/imitate_5_model.h5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_digit5_acc',
    verbose=1,
    save_best_only=True,
    mode='max')
earlystop = EarlyStopping(
    monitor='val_digit5_acc', patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
# model.fit(
#     # train_data,
#     # train_label,
#     # batch_size=400,
#     epochs=100,
#     verbose=2,
#     # validation_data=(vali_data, vali_label),
#     validation_data=vali,
#     callbacks=callbacks_list)

# model.fit_generator(
#     train,
# train_data,
# train_label,
# batch_size=400,
# epochs=100,
# steps_per_epoch=50000,
# verbose=2,
# validation_data=(vali_data, vali_label),
# validation_data=vali,
# validation_steps=10240,
# callbacks=callbacks_list)
model.fit_generator(
    generator=training_datagen,
    validation_data=validation_datagen,
    use_multiprocessing=True,
    callbacks=callbacks_list)

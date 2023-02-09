from __future__ import print_function, division


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import nltk
import numpy as np
from nltk.corpus import treebank
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

nltk.download('treebank')
sentences = nltk.corpus.treebank.tagged_sents()
tags = set([
    tag for sentence in treebank.tagged_sents()
    for _, tag in sentence
])

train_test_cutoff = int(.80 * len(sentences))
training_sentences = sentences[:train_test_cutoff]
testing_sentences = sentences[train_test_cutoff:]
train_val_cutoff = int(.25 * len(training_sentences))
validation_sentences = training_sentences[:train_val_cutoff]
training_sentences = training_sentences[train_val_cutoff:]

def add_basic_features(sentence_terms, index):
    """ Compute some very basic word features.
        :param sentence_terms: [w1, w2, ...]
        :type sentence_terms: list
        :param index: the index of the word
        :type index: int
        :return: dict containing features
        :rtype: dict
    """
    term = sentence_terms[index]
    return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'is_capitalized': term[0].upper() == term[0],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'prefix-3': term[:3],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'suffix-3': term[-3:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }

def untag(tagged_sentence):
    """
    Remove the tag for each tagged term.
:param tagged_sentence: a POS tagged sentence
    :type tagged_sentence: list
    :return: a list of tags
    :rtype: list of strings
    """
    return [w for w, _ in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    """
    Split tagged sentences to X and y datasets and append some basic features.
:param tagged_sentences: a list of POS tagged sentences
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return:
    """
    X, y = [], []
    for pos_tags in tagged_sentences:
        for index, (term, class_) in enumerate(pos_tags):
            # Add basic NLP features for each sentence term
            X.append(add_basic_features(untag(pos_tags), index))
            y.append(class_)
    return X, y

X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(testing_sentences)
X_val, y_val = transform_to_dataset(validation_sentences)

# Fit our DictVectorizer with our set of features
dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train)
dict_vectorizer.fit(X_test)
dict_vectorizer.fit(X_val)



# Convert dict features to vectors
X_train = dict_vectorizer.transform(X_train)
X_test = dict_vectorizer.transform(X_test)
# X_val = dict_vectorizer.transform(X_val)

# Fit LabelEncoder with our list of classes
label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_val)
# Encode class values as integers
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

X_train_set=X_train[0:1]
y_train_set=y_train[0:1]


class ACGAN():
    def __init__(self):
        # Input shape
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channels = 1
        self.img_shape = 17781
        self.num_classes = 46
        self.latent_dim = 23552

        optimizer = Adam(0.0002, 0.5)
        losses = ['categorical_crossentropy', 'categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(46,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):
        model = Sequential([
            Dense(512, input_dim=self.latent_dim),
            Activation('relu'),
            Dropout(0.2),
            Dense(512),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.img_shape, activation='relu')
        ])

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(46,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 512)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential([
            Dense(512, input_dim=self.img_shape),
            Activation('relu'),
            Dropout(0.2),
            Dense(512),
            Activation('relu'),
            Dropout(0.2),

        ])
        model.summary()

        img = Input(shape=(self.img_shape,))

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(46, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs,batch_size=256):



        # Adversarial ground truths
        valid = np.ones((batch_size, 46))
        fake = np.zeros((batch_size, 46))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

         # Select a random batch of data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generated Labels
            sampled_labels = np.random.randint(0, 1, (batch_size, 46))

            # Generate new data
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-45 if tags are valid or 46 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = 46 * np.ones(img_labels.shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

    #         # If at save interval => save generated image samples
    #         if epoch % sample_interval == 0:
    #             self.save_model()
    #             self.sample_images(epoch)
    #
    # def sample_images(self, epoch):
    #     r, c = 10, 10
    #     noise = np.random.normal(0, 1, (r * c, 100))
    #     sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    #     gen_imgs = self.generator.predict([noise, sampled_labels])
    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%d.png" % epoch)
    #     plt.close()
    #
    # def save_model(self):
    #
    #     def save(model, model_name):
    #         model_path = "saved_model/%s.json" % model_name
    #         weights_path = "saved_model/%s_weights.hdf5" % model_name
    #         options = {"file_arch": model_path,
    #                     "file_weight": weights_path}
    #         json_string = model.to_json()
    #         open(options['file_arch'], 'w').write(json_string)
    #         model.save_weights(options['file_weight'])
    #
    #     save(self.generator, "generator")
    #     save(self.discriminator, "discriminator")


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=100,batch_size=256)

#Checking Accuracy
X_validation, y_validation = transform_to_dataset(training_sentences)
y_categorical_values=[]
for val in y_train:
    value=np.where(val==1)
    value=value[0][0]
    y_categorical_values.append(value)

y_categorical_values=np.array(y_categorical_values)
y_categorical_tags=np.array(y_validation)
categorical_tags_values={'tag':y_categorical_tags,'value':y_categorical_values}
categorical_tags_values=pd.DataFrame(categorical_tags_values)
categorical_tags_values=categorical_tags_values.drop_duplicates()

prediction_array=acgan.discriminator.predict(X_train)
prediction_array=np.array(prediction_array)
prediction_array=prediction_array[1]

predicted_tag_list=[]
for pred in prediction_array:
    predicted=np.where(pred==max(pred))
    predicted=predicted[0][0]
    predicted_tag_list.append(predicted)

original_tag_list=[]
for org in y_train:
    original=np.where(org==1)
    original=original[0][0]
    original_tag_list.append(original)

correct = 0
for i in range(len(original_tag_list)):
	if original_tag_list[i] == predicted_tag_list[i]:
	   correct += 1

print('Accuracy (%)',correct/len(original_tag_list)*100)

#predicting a random training data
prediction=acgan.discriminator.predict(X_train[2:3])
prediction_tag=np.where(sum(prediction[1])==max(sum(prediction[1])))
prediction_tag=prediction_tag[0][0]
prediction_tag=int(prediction_tag)
predicted_tag=categorical_tags_values.ix[(categorical_tags_values['value']==prediction_tag),'tag']
predicted_tag=np.array(predicted_tag)[0]
original_tag=y_validation[2:3]

print('Predicted Tag',predicted_tag,'Original Tag',original_tag,'Original Features',X_validation[2:3])


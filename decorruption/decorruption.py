# General module
import os
import argparse
import shutil
import operator
import numpy as np
from random import randint
from tqdm import tqdm

# OpenCV
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, resize

# Keras
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Decorruption:

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--deconstruct', type=str, default=None, help='Deconstruction: from video to frames')
        self.parser.add_argument('--get_features', nargs='*', default=None, help='Reconstruction: from frames to video' )
        self.parser.add_argument('--reduce_dim', nargs='*', default=None, help='reduce dimensionality' )
        self.parser.add_argument('--decorruption', type=str, default=None, help='All pipeline at once')

        self.args = self.parser.parse_args()

        # Path
        self.frames_path = 'data/input/frames/'
        self.output_frames = 'data/output/frames/'
        self.output_video = 'data/output/'
        self.saved_features = 'data/saved/'
        # Mode
        self.TEST_MODE_TEST = 'get_features'
        self.REDUCE_DIM = 'reduce_dim'
        self.DECORRUPT = 'decorruption'

        self.list_frame = []
        self.nb_frames = 0
        self.model = None
        self.list_features = []
        self.closest_index = []
        self.list_path_frames = []

    def main(self, argv=None):

        # All pipeline
        # Ensure that bool(self.args.decorruption) is True even if decorruption == []
        if self.args.decorruption is not None and not self.args.decorruption:
            self.args.decorruption = [self.DECORRUPT]  # Default mode

        if self.args.decorruption:
            # deconstruct the video, get all the frames
            self._deconstruct(self.args.decorruption)
            assert os.path.exists("data/input/frames/frame_1.jpg")

            # Load all the frames
            self.list_frame, self.list_path_frames = self._load_imgtoarray(self.frames_path)

            # Get the frame's features
            self.model = self._model()
            self.list_features = self._get_features(self.list_frame)

            # Reduce dimensionality
            scaler = StandardScaler()
            scaler.fit(self.list_features)
            train_img = scaler.transform(self.list_features)
            pca = PCA(.95)
            pca.fit(train_img)
            train_img = pca.transform(train_img)
            # print(train_img.shape)

            # Get nearest neighbor
            indices, distances = self._get_3_closest(train_img)

            # Get ordered list
            linked_list = self._get_linked_list(indices, distances)

            # Create final video
            self._reorganized(linked_list, self.list_frame)

        # Deconstruct the video: From video to individual frames
        if(self.args.deconstruct != None):
            self._deconstruct(self.args.deconstruct)

        # Get feature from frames using VGG16
        if self.args.get_features is not None and not self.args.get_features:
            self.args.get_features = [self.TEST_MODE_TEST]  # Default mode

        if self.args.get_features:

            # Check existing frames
            assert os.path.exists("data/input/frames/frame_1.jpg")
            self.list_frame, self.list_path_frames = self._load_imgtoarray(self.frames_path)

            # Load model and get features
            self.model = self._model()
            list_feature = self._get_features(self.list_frame)
            self.list_features = list_feature

            # Save feature into npy file
            # self._save_features(list_feature)
            np.save(os.path.join(self.saved_features,'features.npy'), list_feature)

        # Reduce dimensionality and find a linked list of nearest neighbor vectors
        if self.args.reduce_dim is not None and not self.args.reduce_dim:
            self.args.reduce_dim = [self.REDUCE_DIM]  # Default mode

        if self.args.reduce_dim:

            # Check existing frames
            assert os.path.exists("data/input/frames/frame_1.jpg")
            self.list_frame, self.list_path_frames = self._load_imgtoarray(self.frames_path)

            # Load feature file
            array = np.load(os.path.join(self.saved_features,'features.npy'))

            # Reduce dimensionality through PCA
            scaler = StandardScaler()
            scaler.fit(array)
            train_img = scaler.transform(array)
            pca = PCA(.95)
            pca.fit(train_img)
            train_img = pca.transform(train_img)

            # Find nearest neighbor
            indices, distances = self._get_3_closest(train_img)

            # Find sequence
            linked_list = self._get_linked_list(indices, distances)

            # Re-arrange sequence and export it to video file
            self._reorganized(linked_list, self.list_frame)

    def _deconstruct(self, video_path):
        """
        The video is decomposed into individual frames
        :param video_path:
        :return: the number of frames
        """
        # Get paths
        input_path = os.path.abspath(os.path.join(video_path, os.pardir))
        self.frames_path = os.path.join(input_path,"frames")

        # Create frames folder
        if(os.path.isdir(self.frames_path)):
            shutil.rmtree(self.frames_path)
        os.makedirs(self.frames_path)

        # Extract frames
        video_frames = cv2.VideoCapture(video_path)
        nb_images = 0
        result = True
        while result:
            result, frames = video_frames.read()
            if(result):
                cv2.imwrite(os.path.join(self.frames_path, "frame_{}.jpg".format(nb_images)), frames)
                nb_images += 1

    def _get_features(self, lst_files):
        """
        For all frames, extract feature vector from VGG16
        :param lst_files:
        :return: lst_features
        """

        lst_features = [] # list of file_feature

        # Load VGG16 model
        self.model = self._model()

        for img_file in tqdm(lst_files, desc='Extracting features'):

            # Resize img in order to fit the required input dimension for VGG
            img = cv2.resize(img_file, (224, 224))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Get the feature vector regarding the specified layer in _model()
            features = self.model.predict(x)

            lst_features.append(features[0])

        return lst_features

    def _model(self):
        """

        :return: VGG16 model regarding a specific layer, here fc1
        """
        # Load global model VGG16
        model_global = keras.applications.vgg16.VGG16(include_top=True,
                                                      weights='imagenet',
                                                      input_tensor=None,
                                                      input_shape=None,
                                                      pooling=None,
                                                      classes=1000)
        # Create model from specific layer: fc1
        layer_model = keras.models.Model(inputs=model_global.input,
                                         outputs=model_global.get_layer('fc1').output)
        return layer_model

    def _load_imgtoarray(self,pth):
        """

        :param pth:
        :return: list of loaded frames and their corresponding path
        """
        lst_frame = []
        lst_path = []
        for file in tqdm(os.listdir(pth), desc='Loading img'):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(pth,file))
                lst_path.append(os.path.join(pth,file))
                smaller_img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
                lst_frame.append(smaller_img)

        self.nb_frames = len(lst_frame)
        return lst_frame, lst_path

    def _get_3_closest(self, lst_vect):
        """
        Get a list of nearest neighbor for each vector
        :param lst_vect:
        :return:
        """
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(lst_vect)
        distances, indices = nbrs.kneighbors(lst_vect)
        return indices, distances

    def _get_linked_list(self, indices, distances):
        """
        Try to find a logic sequence into the data, first try to extract one side on the "linked list" and then the second side.
        :param indices: index of closest vectors
        :param distances: distances of each closest vector
        :return: a sequence of frame index
        """
        lst_sequence = []
        len_counter = {}

        for i in tqdm(range(100), desc='Iteration'):
            first_pass = []
            second_pass = []

            current = randint(0, len(indices)-1)
            first_pass.append(current)
            first_pass_1 = 1
            while first_pass_1:
                if indices[current][1] not in first_pass:
                    first_pass.append(indices[current][1])
                elif indices[current][2] not in first_pass:
                    first_pass.append(indices[current][2])
                elif indices[current][3] not in first_pass:
                    first_pass.append(indices[current][3])
                elif indices[current][4] not in first_pass:
                    first_pass.append(indices[current][4])
                elif indices[current][5] not in first_pass:
                    first_pass.append(indices[current][5])
                else:
                    first_pass_1 = 0
                current = first_pass[-1]

            second_pass.append(first_pass[0])
            second_pass_1 = 1
            current = second_pass[0]

            while second_pass_1:
                if indices[current][1] not in first_pass and indices[current][1] not in second_pass:
                    second_pass.append(indices[current][1])
                elif indices[current][2] not in first_pass and indices[current][2] not in second_pass:
                    second_pass.append(indices[current][2])
                elif indices[current][3] not in first_pass and indices[current][3] not in second_pass:
                    second_pass.append(indices[current][3])
                elif indices[current][4] not in first_pass and indices[current][4] not in second_pass:
                    second_pass.append(indices[current][4])
                elif indices[current][5] not in first_pass  and indices[current][5] not in second_pass:
                    second_pass.append(indices[current][5])
                else:
                    second_pass_1 = 0
                current = second_pass[-1]

            sequence = second_pass[::-1] + first_pass

            if len(sequence) not in list(len_counter.keys()):
                len_counter[len(sequence)] = 1
                lst_sequence.append(sequence)
            else:
                len_counter[len(sequence)] += 1

        most_common_len = max(len_counter.items(), key=operator.itemgetter(1))[0]
        print("The final sequence has {} frames".format(most_common_len))

        sequence_final = [x for x in lst_sequence if len(x) == most_common_len]

        # print("final sequence: {}".format(sequence_final))
        return sequence_final[0]

    def _reorganized(self, list_indices, list_img):
        """Reorder the sequence to have the frames instead of the index"""
        ordered_img = []
        for i, indice in enumerate(list_indices):
            img = list_img[indice]
            ordered_img.append(img)
            cv2.imwrite(os.path.join(self.output_frames,"img{}.jpg".format(i)), img)
        self._make_video(ordered_img)

    def _make_video(self, images, outimg=None, fps=24, size=None, is_color=True, format="XVID"):
        """From frames to video file"""
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for img in tqdm(images, desc='Creating video'):
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(os.path.join(self.output_video,"result.avi"), fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        vid = None
        for img in tqdm(images[::-1], desc='Creating reversed video'):
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(os.path.join(self.output_video,"result_reversed.avi"), fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid
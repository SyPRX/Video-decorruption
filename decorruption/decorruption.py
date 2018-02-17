import os
import argparse
import cv2
import shutil
import tensorflow
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randint



# Import for tsne
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
# from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# %matplotlib inline

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
from sklearn.mixture import GMM


class Decorruption:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--deconstruct', type=str, default=None, help='Deconstruction: from video to frames')
        self.parser.add_argument('--reconstruct', nargs='*', default=None, help='Reconstruction: from frames to video' )
        self.parser.add_argument('--optical_flow', nargs='*', default=None, help='Use optical flow techniques to find similarity between images' )
        self.parser.add_argument('--rm_outlier', nargs='*', default=None, help='Remove outlier images in order to delete the corrupt data' )
        self.parser.add_argument('--reduce_dim', nargs='*', default=None, help='reduce dimensionality' )
        self.parser.add_argument('--color_hist', nargs='*', default=None, help='Color Hist for images')
        self.args = self.parser.parse_args()

        self.frames_path = "data/input/frames/"
        self.TEST_MODE_TEST = 'reconstruct'
        self.REDUCE_DIM = 'reduce_dim'
        self.COLOR_HIST = 'color_hist'

        self.list_frame = []
        self.nb_frames = 0
        self.model = None
        self.list_features = []
        self.closest_index = []

        self.list_path_frames = []
    def main(self, argv=None):

        if(self.args.deconstruct != None):
            self._deconstruct(self.args.deconstruct)

        # Check if frame directory not empty
        assert os.path.exists("data/input/frames/frame_1.jpg")

        # Get list of frames

        self.list_frame, self.list_path_frames = self._load_imgtoarray(self.frames_path)

        # Ensure that bool(self.args.reconstruct) is True even if reconstruct == []
        if self.args.color_hist is not None and not self.args.color_hist:
            self.args.color_hist = [self.COLOR_HIST]  # Default mode

        if self.args.color_hist:

            self._img2hist(self.list_frame)



        # if self.args.rm_outlier:
        #     pass
        # Ensure that bool(self.args.reconstruct) is True even if reconstruct == []
        if self.args.reduce_dim is not None and not self.args.reduce_dim:
            self.args.reduce_dim = [self.REDUCE_DIM]  # Default mode

        print("Before")
        if self.args.reduce_dim:
            print("Load file")
            array = np.load('testnp.npy')
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Fit on training set only.
            scaler.fit(array)
            # Apply transform to both the training set and the test set.
            train_img = scaler.transform(array)

            from sklearn.decomposition import PCA
            # Make an instance of the Model
            pca = PCA(.95)
            pca.fit(train_img)
            train_img = pca.transform(train_img)
            print(train_img.shape)
            # self._GMM(train_img)

            # import matplotlib.pyplot as plt
            # plt.plot(train_img[:,0], train_img[:,1], 'ro')
            # plt.show()
            # plt.plot(train_img[:, 1], train_img[:, 2], 'ro')
            # plt.show()
            # plt.plot(train_img[:, 0], train_img[:, 2], 'ro')
            # plt.show()
            indices, distances = self._get_3_closest(train_img)
            print("here2")

            linked_list = self._get_linked_list(indices, distances)
            print("here1")
            self._reorganized(linked_list, self.list_frame)

        # Ensure that bool(self.args.reconstruct) is True even if reconstruct == []
        if self.args.reconstruct is not None and not self.args.reconstruct:
            self.args.reconstruct = [self.TEST_MODE_TEST]  # Default mode

        if self.args.reconstruct:
            self.model = self._model()
            list_feature = self._get_features(self.list_frame)
            self.list_features = list_feature

            self._save_features(list_feature)
            print(self.list_features[0])
            print(len(self.list_features))
            print(len(self.list_features[0]))
            # print(len(self.list_features[0][0]))
            np.save('testnp.npy', list_feature)
            array = np.load('testnp.npy')
            print(len(array))
            print(len(array[0]))
            print(array[0])
            # print("diff = {}".format(sum(self.list_features[0][0][0]-self.list_features[1][0][0])))
            # print(len(self.list_features[0][1][0][0]))
            # print(len(self.list_features[0][1][0][0][0]))

            # print(self._find_2_nn(list_feature))

        if self.args.optical_flow:
            pass

    def _deconstruct(self, video_path):
        """
        The video is decomposed into multiple frames
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

        return nb_images

    def _get_features(self, lst_files):

        file_feature = [] # feature vector + filename
        lst_features = [] # list of file_feature

        self.model = self._model()

        for img_file in lst_files:
            # print(img_file.shape)
            # file_feature.append(img_file)

            # img = image.load_img(filename, target_size=(224, 244))
            img = cv2.resize(img_file, (224, 224))
            print(img.shape)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            features = self.model.predict(x)
            # print(len(features))
            # self._print_list(features)
            print("type: {}".format(type(features)))
            print(features)
            # print("shape: {}".format(features.shape))
            # print("sum = {}".format(sum(features[0])))
            # file_feature.append(features)
            lst_features.append(features[0])
            # print(len(features[0]))
            # print("features = {}".format(features))
            # Preprocess each frame

        return lst_features



    def _model(self):
        model_global = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                 input_tensor=None, input_shape=None, pooling=None,
                                                                 classes=1000)
        # layer_model = keras.models.Model(inputs=model_global.input, outputs=model_global.get_layer('conv_7b').output)
        model_global = keras.applications.vgg16.VGG16(include_top=True,
                                                      weights='imagenet',
                                                      input_tensor=None,
                                                      input_shape=None,
                                                      pooling=None,
                                                      classes=1000)
        layer_model = keras.models.Model(inputs=model_global.input, outputs=model_global.get_layer('fc1').output)

        return layer_model

    def _save_features(self, lst_features):

        thefile = open('data/saved/lst_features.txt', 'w')
        for item in lst_features:
            thefile.write("{}\n".format(item))
        thefile.close()
        return -1

    def _print_list(self, lst):
        try:
            if(len(lst[0]) == 1):
                return 0
            else:
                for lst_1 in lst:
                    print(len(lst_1))
                    self._print_list(lst_1)
                    return 0
        except:
            pass

    def _find_2_nn(self, lst_f):
        lll = []
        for frame in lst_f:
            lll.append(frame[1])

        X = np.array(lll).flatten().reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        return indices

    def _get_optical_flow(self, img1, img2):
        pass

    def _load_imgtoarray(self,pth):

        lst_frame = []
        lst_path = []
        for file in os.listdir(pth):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(pth,file))
                lst_path.append(os.path.join(pth,file))
                smaller_img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
                lst_frame.append(smaller_img)

                # print("shape img {}".format(smaller_img.shape))
        # print(len(list_frame))
        self.nb_frames = len(lst_frame)
        print("{} imgs are loaded".format(self.nb_frames))
        return lst_frame, lst_path


    def _t_sne(self, data):
        pass


    def _get_3_closest(self, lst_vect):
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(lst_vect)
        distances, indices = nbrs.kneighbors(lst_vect)
        print("indices : {}".format(indices[0:2]))
        print("distances : {}".format(distances[0:2]))
        return indices, distances

    def _get_linked_list(self, indices, distances):
        first_pass = []
        second_pass = []
        linked_list_index = []

        current = randint(0, len(indices))
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
        print("len first = {}".format(len(first_pass)))
        print("len second = {}".format(len(second_pass)))

        return second_pass[::-1] + first_pass


    def _GMM(self, eigen):
        gmm = GMM(n_components=4, covariance_type='full', random_state=42).fit(eigen)
        # gmm = GMM(n_components=2).fit(eigen)
        labels = gmm.predict(eigen)
        plt.scatter(eigen[:, 0], eigen[:, 1], c=labels, s=40, cmap='viridis')
        plt.show()
        print(labels)
        probs = gmm.predict_proba(eigen)
        print(probs[:5].round(1))

    def _img2hist(self, lst_imgs):

        for img in lst_imgs:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    def _reorganized(self, list_indices, list_img):
        ordered_img = []
        for i, indice in enumerate(list_indices):
            # print("indice = {}".format(indice))
            img = list_img[indice]
            # print(type(img))
            # print(img.shape)
            ordered_img.append(img)
            cv2.imwrite("data/output/img{}.jpg".format(i), img)
        self._make_video(ordered_img)

    def _make_video(self, images, outimg=None, fps=24, size=None, is_color=True, format="XVID"):
        """
        Create a video from a list of images.

        @param      outvid      output video
        @param      images      list of images to use in the video
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

        The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
        By default, the video will have the size of the first image.
        It will resize every image to this size before adding them to the video.
        """
        from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for img in images:
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter("video_1.avi", fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        vid = None
        for img in images[::-1]:
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter("video_2.avi", fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid
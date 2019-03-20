# Copyright 2018 Douglas Feitosa Tome All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import pickle
import json
import gzip
import shutil
import numpy as np
import cv2
import h5py

import vggish_params

def save_pickled_object(object, path):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(object, protocol=4)
    n_bytes = sys.getsizeof(bytes_out)
    with open(path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def load_pickled_object(path):
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(path)
        bytes_in = bytearray(0)
        with open(path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        object = pickle.loads(bytes_in)
    except OSError as err:
        print('OS error: {0}'.format(err))
    except:
        print('Unexpected error:', sys.exc_info()[0])
        raise
    return object

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def unzip_files(source, destination):
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f)) and f.split('.')[-1] == 'gz']
    print('files', len(files))

    for file in files:
        with gzip.open(os.path.join(source, file), 'rb') as f_in:
            with open(os.path.join(destination, '.'.join(file.split('.')[:-1])), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def show_img_cv2(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prepare_image(img_path):
    # Load a color image in grayscale
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (vggish_params.NUM_BANDS, vggish_params.NUM_FRAMES), interpolation=cv2.INTER_CUBIC)
    return img

def load_svhn_data(data_path, set):
    print('Loading data from', os.path.join(data_path, set, 'digitStruct.mat'))
    f = h5py.File(os.path.join(data_path, set, 'digitStruct.mat'), 'r')
    name = f['digitStruct']['name']
    bbox = f['digitStruct']['bbox']
    images = []
    labels = []
    for i in range(name.shape[0]):
        # Retrieve image
        name_array = f[name[i][0]].value
        image_file = ''.join(chr(name_array[j, 0]) for j in range(name_array.shape[0]))
        image = prepare_image(os.path.join(data_path, set, image_file))
        if image.shape == (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS):
            images.append(image)
            # Retrieve label
            label_array = f[bbox[i][0]]['label'].value
            label = np.zeros(10)
            if label_array.shape == (1, 1):
                l = int(label_array[0, 0])
                if l == 10:
                    label[0] = 1
                else:
                    label[l] = 1
            else:
                for l in [int(f[label_array[j, 0]].value[0, 0]) for j in range(label_array.shape[0])]:
                    if l == 10:
                        label[0] = 1
                    else:
                        label[l] = 1
            labels.append(label)
        else:
            print('Image', image_file, 'has bad shape:', image.shape)
    return (images, labels)

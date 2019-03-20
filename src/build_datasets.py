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

from datasets import *

def main():
    build_mnist(data_path='/my/data/path/MNIST', batch_size=vggish_params.BATCH_SIZE, num_features=1)
    
    build_svhn(data_path='/my/data/path/SVHN', batch_size=vggish_params.BATCH_SIZE, num_features=1)
    
    build_extra_svhn(data_path='/my/data/path/SVHN', batch_size=vggish_params.BATCH_SIZE, num_features=1)
    
    build_cifar100(data_path='/my/data/path/CIFAR100', batch_size=vggish_params.BATCH_SIZE, num_features=4)
    
    build_cifar10(data_path='/my/data/path/CIFAR10', batch_size=vggish_params.BATCH_SIZE, num_features=4)
    
    build_audioset_branches(has_quality_filter=False,
                            min_quality=None,
                            has_depth_filter=True,
                            depths=[2],
                            has_rerated_filter=False,
                            ontology_path='/my/data/path/AudioSet/',
                            config_files_path='/my/data/path/AudioSet/',
                            data_path='/my/data/path/AudioSet/',
                            batch_size=vggish_params.BATCH_SIZE)
        
    build_esc50(data_path='/my/data/path/ESC-50', train_folds=[1, 5, 2], validation_fold=[4], test_fold=[3], batch_size=vggish_params.BATCH_SIZE, num_features=4)
    
    build_urbansed(data_path='/my/data/path/URBAN-SED', batch_size=vggish_params.BATCH_SIZE, num_features=4)

if __name__ == '__main__':
    main()


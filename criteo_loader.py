import os
import random
import time
import numpy as np
from itertools import chain
from tqdm import tqdm
#DATA_PATH = 'data/dac_sample/dac_sample.txt'
import pickle
import os

def _numerical_index_mapper(str_val):
    if str_val == '':
        return 0
    else:
        return 1

class CriteoLoader():
    def __init__(self, train_path, test_path, use_numeric=True):
        self.use_numeric = use_numeric
        self.train_path = train_path
        self.test_path = test_path
        self.col_types = ['num'] * 13 + ['categ'] * 26
        self.index_mappers = []
        self._init_index_mappers()

    def _init_index_mappers(self,):
        for no, ctype in enumerate(self.col_types):
            if ctype == 'num':
                if self.use_numeric:
                    self.index_mappers.append(_numerical_index_mapper)
            else:
                name = 'categ{}'.format(no)
                self.index_mappers.append(_Mapper(name))   

    def save_index_mappers(self,):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        with open('temp/index_mappers.pkl', 'wb') as f:
            pickle.dump(self.index_mappers, f)
    
    def load_index_mapper(self, path='temp/index_mappers.pkl'):
        try:
            with open(path, 'rb') as f:
                mappers = pickle.load(f)
            self.index_mappers = mappers
        except FileNotFoundError:
            # no index_mapper found
            self.fit_index_mappers()

    def fit_index_mappers(self, test_also=True):
        'test_also: also create index for category in test_file'
        skip_numeric = 0 if self.use_numeric else 13
        with open(self.train_path, 'rt') as f:
            for l in tqdm(f):
                row = l.strip('\n').split('\t')
                zip_ = zip(self.index_mappers[(13-skip_numeric):], row[14:])
                _ = [map_(feat) for map_, feat in zip_]        
        if test_also:
            print('also load testing file...')
            with open(self.test_path, 'rt') as f:
                for l in tqdm(f):
                    row = l.strip('\n').split('\t')
                    zip_ = zip(self.index_mappers[(13-skip_numeric):], row[14:])
                    _ = [map_(feat) for map_, feat in zip_]        

        self.freeze_mappers()    
        self.save_index_mappers()        
    
    def freeze_mappers(self,):
        for mapper in self.index_mappers:
            if isinstance(mapper, _Mapper):
                mapper.freeze_mapper()
    
    def _generator(self, path): #-> (List[int], List[float], int)
        with open(path, 'rt') as f:
            for l in f:
                row = l.strip('\n').split('\t')
                label = int(row[0])
                features = row[1:]
                if self.use_numeric:
                    indice = [map_(val) for map_, val in zip(self.index_mappers, features)]
                    coefs = [float(i) if i is not '' else 0 for i in features[0:13]] #numerical part
                    categ_coefs = [1] * len(features[13:])
                    coefs.extend(categ_coefs)
                else:
                    indice = [map_(val) for map_, val in zip(self.index_mappers, features[13:])]
                    coefs = [1] * len(features[13:])
                yield indice, coefs, label

    def train_generator(self,):
        return self._generator(self.train_path)
    
    def test_generator(self,):
        return self._generator(self.test_path)

    def get_categ_count(self,):
        return [
            len(mapper._val_index_dic) + 1 if isinstance(mapper, _Mapper) else 2
            for mapper in self.index_mappers
        ]

class _Mapper():    
    def __init__(self, name=None, keep_0_as_padding=True, **meta):        
        self.name = name
        self.meta = meta
        self._freeze = False
        self._val_index_dic = dict()
        self._index_val_dic = dict()
        self.pad_0 = int(keep_0_as_padding)

    def __call__(self, val, handle_invalid='keep'):
        return self._get_index_mapping(val, handle_invalid)

    def _get_index_mapping(self, val, handle_invalid):
        if val not in self._val_index_dic and not self._freeze:
            index = len(self._val_index_dic) + self.pad_0
            self._val_index_dic[val] = index
            return index
        elif val not in self._val_index_dic and handle_invalid == 'keep':
            return self._val_index_dic.get(val, 0)
        else:
            return self._val_index_dic[val]   

    def get_val_mapping(self, index):
        if len(self._val_index_dic) != len(self._index_val_dic):
            self._index_val_dic = {val:idx for idx, val in self._val_index_dic.items()}
        return self._index_val_dic[index]
        
    def freeze_mapper(self):
        'not allow adding new value. Any new value to Mapper will raise keyerror'
        self._freeze = True

    def unfreeze_mapper(self):
        self._freeze = False


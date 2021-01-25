import os
import random
import time
import numpy as np
#import abc
from itertools import chain
from loader import _Loader
from ml100k_loader import Movie, UserInfoLoader, _pad_0, ML100kLoader

base_path = '/Users/ericlai/Documents/py/try_pruning/data/ml-1m/'
USER_PATH = os.path.join(base_path, 'users.dat')
MOVIE_PATH = os.path.join(base_path, 'movies.dat')
TRAIN_RATING_PATH = os.path.join(base_path, 'rating_train0.8.dat')
VALIDATION_RATING_PATH = os.path.join(base_path, 'rating_test0.2.dat')

class ML1MLoader(ML100kLoader):
    '''
    train_rating_generator and val_rating_generator will return a generator for 
    data iteration and tensorflow dataset creation.
    A iter will contain tuple (x, y). x is list of shape: [num_fields, values]. y is a int value.
    '''
    def __init__(self,):
        #self.use_movie_info = use_movie_info
        #self.use_user_info = use_user_info
        self._train_ratings = [i for i in self._load_rating_generator(TRAIN_RATING_PATH)]
        self._val_ratings =  [i for i in self._load_rating_generator(VALIDATION_RATING_PATH)]
        self.user_info_loader = _UserInfoLoader()
        self.user_info_loader.load(USER_PATH)
        self.movie_info_loader = _MovieInfoLoader()
        self.movie_info_loader.load(MOVIE_PATH)


    def _load_rating_generator(self, path):
        with open(path, 'rt') as f:
            for l in f:
                uid, iid, rate, _ = l.strip().split("::")
                uid = int(uid)
                iid = int(iid)
                rate = int(rate)
                yield [uid, iid, rate]                


class _MovieInfoLoader(_Loader):
    'load movie information'
    def __init__(self,):
        self._mid_mapper = dict()
        self.genre_map = dict()

    @property
    def _mapper(self):
        return self._mid_mapper
    
    def load(self, path):
        with open(path, 'rt', errors='replace') as f:
            for l in f:
                mid, title, genres = l.strip('\n').split("::")
                mid = int(mid)
                indexed_genres = [
                    self.genre_map.setdefault(g, len(self.genre_map)+1 ) 
                    for g in genres.split('|')
                ]
                indexed_genres = np.array(_to_onehot(indexed_genres))
                self._mid_mapper[int(mid)] = Movie(mid, title, None, None, indexed_genres)


class _UserInfoLoader(UserInfoLoader):
    '''load user information.
       Note that age is transform to discrete ints with [0,10) -> 1, [10,20) -> 2, ..., 
    
    '''
    def __init__(self,):
        super().__init__()
        
    def load(self, path):
        with open(path, 'rt') as f:
            for l in f:                
                #1::F::1::10::48067
                line = l.strip().split('::')
                uid, gender, age, occupation, zipcode = line
                uid = int(uid)
                if uid in self._uid_mapper:
                    pass
                else:
                    feature = self._get_user_feature(age, gender, occupation, zipcode)
                    self._uid_mapper[uid] = feature        
                

def _to_onehot(index, dim=18):
    'note that index start with 1'
    return [
        1 if i+1 in index else 0 
        for i in range(18)
    ]
                    
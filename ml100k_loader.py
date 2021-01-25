import os
import random
import time
import numpy as np
#import abc
from itertools import chain
from loader import _Loader

BASE_PATH = '/Users/ericlai/Documents/py/Rec2.0/data/ml-100k'
USER_PATH = os.path.join(BASE_PATH, 'u.user')
MOVIE_PATH = os.path.join(BASE_PATH, 'u.item')
RATING_PATH = os.path.join(BASE_PATH, 'u.data')
TRAIN_RATING_PATH =  os.path.join(BASE_PATH, 'u1.base')
VALIDATION_RATING_PATH = os.path.join(BASE_PATH, 'u1.test')


class ML100kLoader():
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
        self.user_info_loader = UserInfoLoader()
        self.user_info_loader.load(USER_PATH)
        self.movie_info_loader = MovieInfoLoader()
        self.movie_info_loader.load(MOVIE_PATH)
    
    def train_rating_generator(self, use_movie_info=True, use_user_info=True, 
                               combine_movie_features=True): #-> Generator
        '''
        combine_movie_features: whether to combine genres feature from n binary 
                                feature to 1 multi-class features.
        '''
        generator = self._iter_data(self._train_ratings, use_movie_info, use_user_info)
        if combine_movie_features:
            return ML100kLoader.combine_movie_features(generator)  
        else:      
            #return ML100kLoader.expend_dim_at_last(generator)
            return generator

    def val_rating_generator(self, use_movie_info=True, use_user_info=True,
                             combine_movie_features=True): #-> Generator        
        generator = self._iter_data(self._val_ratings, use_movie_info, use_user_info)
        if combine_movie_features:
            return ML100kLoader.combine_movie_features(generator)  
        else:      
            #return ML100kLoader.expend_dim_at_last(generator)
            return generator

    def _load_rating_generator(self, path):
        with open(path, 'rt') as f:
            for l in f:
                uid, iid, rate, _ = l.strip().split("\t")
                uid = int(uid)
                iid = int(iid)
                rate = int(rate)
                yield [uid, iid, rate]                

    def _iter_data(self, rating_datas, use_movie_info, use_user_info):
        'use_movie_info, use_user_info: Bool. Whether to use movie(user) information data.'
        _get_user = lambda uid: self.user_info_loader[uid] if use_user_info else []
        _get_movie = lambda mid: self.movie_info_loader[mid].genres if use_movie_info else []
        for data in rating_datas:
            X =  list(chain(
                data[0:2], _get_user(data[0]), _get_movie(data[1])
            ))
            Y = data[2]
            yield X, Y
    
    @staticmethod
    def expend_dim_at_last(generator): #->Generator
        for x, y in generator:
            yield [[i] for i in x], y            
    
    @staticmethod    
    def combine_movie_features(generator): #->Generator
        '''combine movie features from n features to 1
           also make data from List[int] to List[List[int]] 
           and pad 0 to make it a tensor-like shape
        '''
        for x, y in generator:
            movie_features = x[-19:]
            movie_features = [no for no, i  in enumerate(movie_features, 1) if i==2]
            features = [_pad_0([i], 19) for i in x[:-19]] #all features expect movie_features
            features.append(_pad_0(movie_features, 19))
            yield features, y    

def _pad_0(li, n):
    return li + [0] * (n- len(li))


class UserInfoLoader(_Loader):
    '''load user information.
       Note that age is transform to discrete ints with [0,10) -> 1, [10,20) -> 2, ..., 
    
    '''
    def __init__(self,):
        self._uid_mapper = dict()
        self._occupation_mapper = dict()
        self._gender_mapper = {'M':1, 'F':2}
        self._zipcode_mapper =  dict()

    @property
    def _mapper(self):
        return self._uid_mapper
        
    def load(self, path):
        with open(path, 'rt') as f:
            for l in f:                
                line = l.strip().split('|')
                uid, age, gender, occupation, zipcode = line
                uid = int(uid)
                if uid in self._uid_mapper:
                    pass
                else:
                    feature = self._get_user_feature(age, gender, occupation, zipcode)
                    self._uid_mapper[uid] = feature        
                
    def _get_user_feature(self, age, gender, occupation, zipcode):
        gender_id = self._gender_mapper[gender]
        if occupation not in self._occupation_mapper:
            self._occupation_mapper[occupation] = len(self._occupation_mapper) + 1
        if zipcode not in self._zipcode_mapper:
            self._zipcode_mapper[zipcode] = len(self._zipcode_mapper) + 1
        occ_id = self._occupation_mapper[occupation]
        zip_id = self._zipcode_mapper[zipcode]        
        ordinal_age = int(age)//10 + 1
        return np.array([ordinal_age, gender_id, occ_id, zip_id])


class MovieInfoLoader(_Loader):
    'load movie information'
    def __init__(self,):
        self._mid_mapper = dict()

    @property
    def _mapper(self):
        return self._mid_mapper
    
    def load(self, path):
        with open(path, 'rb') as f:
            for l in f:                
                line = l.strip().split(b'|')
                mid = int(line[0])
                if mid in self._mid_mapper:
                    pass
                else:
                    movie = self.create_movie_instance(line)
                    self._mid_mapper[mid] = movie
    
    def create_movie_instance(self, line): #-> Movie
        'line: List[str]'
        mid = int(line[0])
        title = line[1] if line[1] else None
        release_date = time.strptime(line[2].decode('utf-8'),'%d-%b-%Y') if line[2] else None
        url = line[4] if line[4] else None
        genres = np.array([int(i) for i in line[5:]])
        return Movie(mid, title, release_date, url, genres)
                                    
                
class Movie():
    def __init__(self, mid, title, release_date, url, genres):
        '''
        args:
        ----
        mid: int. movie id.
        title: bytes. movie title name
        release_date: time.struct_time. the release date.
        url: bytes. url.
        genres: Array[int]. numpy array specify genre.
        '''
        self.mid = mid
        self.title = title
        self.release_date = release_date
        self.url = url
        self.genres = genres
    
    def __repr__(self):
        return '<Movie: {} with mid: {}>'.format(self.title, self.mid)


def train_test_stratified_sampling(items, strat=5, seed=123): #-> (trainSet, testSet)
    'take one sample from each stra as test set'
    #items : Dict of key-value pair. value is used to do stratified_sampling
    train_set = []
    test_set = []
    n = len(items)
    sorted_items = sorted(items.items(), key=lambda x:x[1])
    strat_length = int(n/strat)
    indice = list(range(0, n, strat_length))[:strat]
    indice.append(n)
    for i in range(len(indice)-1):
        start = indice[i]
        end = indice[i+1]
        current_strat = sorted_items[start:end]
        random.seed(seed+i)
        random.shuffle(current_strat)        
        test_set.append(current_strat.pop())
        train_set.extend(current_strat)
    return dict(train_set), dict(test_set)
    
    
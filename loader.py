
import abc

class _Loader(abc.ABC):
    'A data loader which map data with int index.'
    @abc.abstractmethod
    def load(self, path):
        'load the data from file'
        return NotImplemented

    @property
    @abc.abstractmethod
    def _mapper(self):
        'the int_index to original data mapping dict.'
        return NotImplemented

    def __getitem__(self, key):
        return self._mapper[key]
    
    def __len__(self):
        return len(self._mapper)
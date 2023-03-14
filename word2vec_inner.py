import numpy as np
import numpy as __numpy


FAST_VERSION = 0

MAX_WORDS_IN_BATCH = 10000


def score_sentence_cbow(model, sentence, _work, _neu1):  # real signature unknown; restored from __doc__
    pass


def score_sentence_sg(model, sentence, _work):  # real signature unknown; restored from __doc__
    pass


def train_batch_cbow(model, sentences, alpha, _work, _neu1,
                     compute_loss):  # real signature unknown; restored from __doc__
    pass


def train_batch_sg(model, sentences, alpha, _work, compute_loss):  # real signature unknown; restored from __doc__
    pass


class REAL(__numpy.floating):
    def as_integer_ratio(self):  # real signature unknown; restored from __doc__
        pass

    def is_integer(self):  # real signature unknown; restored from __doc__
        return False

    def __abs__(self, *args, **kwargs):  # real signature unknown
        pass

    def __add__(self, *args, **kwargs):  # real signature unknown
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        pass

    @classmethod
    def __class_getitem__(cls, *args, **kwargs):  # real signature unknown
        pass

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        pass

    def __float__(self, *args, **kwargs):  # real signature unknown
        pass

    def __floordiv__(self, *args, **kwargs):  # real signature unknown
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        pass

    def __hash__(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        pass

    def __mul__(self, *args, **kwargs):  # real signature unknown
        """ Return self*value. """
        pass

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        pass

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        pass

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    def __sub__(self, *args, **kwargs):  # real signature unknown
        """ Return self-value. """
        pass

    def __truediv__(self, *args, **kwargs):  # real signature unknown
        """ Return self/value. """
        pass


# variables with complex values

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x000002874A931570>'

__pyx_capi__ = {
    'EXP_TABLE': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t [0x3E8]" at 0x000002874A9315C0>'
    'bisect_left': None,
    # (!) real value is '<capsule object "unsigned PY_LONG_LONG (__pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, unsigned PY_LONG_LONG, unsigned PY_LONG_LONG)" at 0x000002874A931710>'
    'dsdot': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_dsdot_ptr" at 0x000002874A931530>'
    'init_w2v_config': None,
    # (!) real value is '<capsule object "PyObject *(struct __pyx_t_6gensim_6models_14word2vec_inner_Word2VecConfig *, PyObject *, PyObject *, PyObject *, PyObject *, struct __pyx_opt_args_6gensim_6models_14word2vec_inner_init_w2v_config *__pyx_optional_args)" at 0x000002874A931830>'
    'our_dot': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_our_dot_ptr" at 0x000002874A9315F0>'
    'our_dot_double': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x000002874A931650>'
    'our_dot_float': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x000002874A931680>'
    'our_dot_noblas': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x000002874A9316B0>'
    'our_saxpy': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_our_saxpy_ptr" at 0x000002874A931620>'
    'our_saxpy_noblas': None,
    # (!) real value is '<capsule object "void (int const *, float const *, float const *, int const *, float *, int const *)" at 0x000002874A9316E0>'
    'random_int32': None,
    # (!) real value is '<capsule object "unsigned PY_LONG_LONG (unsigned PY_LONG_LONG *)" at 0x000002874A931740>'
    'saxpy': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_saxpy_ptr" at 0x000002874A931410>'
    'scopy': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_scopy_ptr" at 0x000002874A930DE0>'
    'sdot': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_sdot_ptr" at 0x000002874A931470>'
    'snrm2': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_snrm2_ptr" at 0x000002874A9314D0>'
    'sscal': None,
    # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_sscal_ptr" at 0x000002874A931590>'
    'w2v_fast_sentence_cbow_hs': None,
    # (!) real value is '<capsule object "void (__pyx_t_5numpy_uint32_t const *, __pyx_t_5numpy_uint8_t const *, int *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int, int, int, int, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_5numpy_uint32_t const , int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x000002874A9317D0>'
    'w2v_fast_sentence_cbow_neg': None,
    # (!) real value is '<capsule object "unsigned PY_LONG_LONG (int const , __pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, int *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int, int, int, int, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_5numpy_uint32_t const , int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x000002874A931800>'
    'w2v_fast_sentence_sg_hs': None,
    # (!) real value is '<capsule object "void (__pyx_t_5numpy_uint32_t const *, __pyx_t_5numpy_uint8_t const *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_5numpy_uint32_t const , int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x000002874A931770>'
    'w2v_fast_sentence_sg_neg': None,
    # (!) real value is '<capsule object "unsigned PY_LONG_LONG (int const , __pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const , __pyx_t_5numpy_uint32_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_5numpy_uint32_t const , int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x000002874A9317A0>'
}

__spec__ = None  # (!) real value is "ModuleSpec(name='gensim.models.word2vec_inner', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x000002874A931570>, origin='C:\\\\Users\\\\Hamza\\\\PycharmProjects\\\\HTML topic model\\\\venv\\\\lib\\\\site-packages\\\\gensim\\\\models\\\\word2vec_inner.cp310-win_amd64.pyd')"

__test__ = {}


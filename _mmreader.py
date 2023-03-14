# encoding: utf-8
# module gensim.corpora._mmreader
# from C:\Users\Hamza\Git projects\WebTopicModel-Prototype\venv\lib\site-packages\gensim\corpora\_mmreader.cp310-win_amd64.pyd
# by generator 1.147
""" Reader for corpus in the Matrix Market format. """

# imports
import builtins as __builtins__  # <module 'builtins' (built-in)>
import logging as logging  # C:\Users\Hamza\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py
import utils as utils  # C:\Users\Hamza\PycharmProjects\HTML topic model\venv\lib\site-packages\gensim\utils.py


# functions

def __pyx_unpickle_MmReader(__pyx_type, long___pyx_checksum,
                            __pyx_state):  # real signature unknown; restored from __doc__
    """ __pyx_unpickle_MmReader(__pyx_type, long __pyx_checksum, __pyx_state) """
    pass


# classes

class MmReader(object):
    """
    MmReader(input, transposed=True)
    Matrix market file reader (fast Cython version), used internally in :class:`~gensim.corpora.mmcorpus.MmCorpus`.

        Wrap a term-document matrix on disk (in matrix-market format), and present it
        as an object which supports iteration over the rows (~documents).

        Attributes
        ----------
        num_docs : int
            Number of documents in the market matrix file.
        num_terms : int
            Number of terms.
        num_nnz : int
            Number of non-zero terms.

        Notes
        -----
        Note that the file is read into memory one document at a time, not the whole matrix at once
        (unlike e.g. `scipy.io.mmread` and other implementations).
        This allows us to process corpora which are larger than the available RAM.
    """

    def docbyoffset(self, offset):  # real signature unknown; restored from __doc__
        """
        MmReader.docbyoffset(self, offset)
        Get the document at file offset `offset` (in bytes).

                Parameters
                ----------
                offset : int
                    File offset, in bytes, of the desired document.

                Returns
                ------
                list of (int, str)
                    Document in sparse bag-of-words format.
        """
        pass

    def skip_headers(self, input_file):  # real signature unknown; restored from __doc__
        """
        MmReader.skip_headers(self, input_file)
        Skip file headers that appear before the first document.

                Parameters
                ----------
                input_file : iterable of str
                    Iterable taken from file in MM format.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        """
        Parameters
                ----------
                input : {str, file-like object}
                    Path to the input file in MM format or a file-like object that supports `seek()`
                    (e.g. smart_open objects).

                transposed : bool, optional
                    Do lines represent `doc_id, term_id, value`, instead of `term_id, doc_id, value`?
        """
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        """
        Iterate through all documents in the corpus.

                Notes
                ------
                Note that the total number of vectors returned is always equal to the number of rows specified
                in the header: empty documents are inserted and yielded where appropriate, even if they are not explicitly
                stored in the Matrix Market file.

                Yields
                ------
                (int, list of (int, number))
                    Document id and document in sparse bag-of-words format.
        """
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        """ Get the corpus size: total number of documents. """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        """ MmReader.__reduce_cython__(self) """
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        """ MmReader.__setstate_cython__(self, __pyx_state) """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    input = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """input: object"""

    num_docs = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_docs: 'long long'"""

    num_nnz = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_nnz: 'long long'"""

    num_terms = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_terms: 'long long'"""

    transposed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """transposed: 'bool'"""


# variables with complex values

logger = None  # (!) real value is '<Logger gensim.corpora._mmreader (DEBUG)>'

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x0000015D7A4F2230>'

__spec__ = None  # (!) real value is "ModuleSpec(name='gensim.corpora._mmreader', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x0000015D7A4F2230>, origin='C:\\\\Users\\\\Hamza\\\\PycharmProjects\\\\HTML topic model\\\\venv\\\\lib\\\\site-packages\\\\gensim\\\\corpora\\\\_mmreader.cp310-win_amd64.pyd')"

__test__ = {}


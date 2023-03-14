#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Basic interfaces used across the whole Gensim package.

These interfaces are used for building corpora, model transformation and similarity queries.

The interfaces are realized as abstract base classes. This means some functionality is already
provided in the interface itself, and subclasses should inherit from these interfaces
and implement the missing methods.

"""

import logging

import utils, matutils


logger = logging.getLogger(__name__)


class CorpusABC(utils.SaveLoad):
    def __iter__(self):
        """Iterate all over corpus."""
        raise NotImplementedError('cannot instantiate abstract base class')

    def save(self, *args, **kwargs):
        """Saves the in-memory state of the corpus (pickles the object).

        Warnings
        --------
        This saves only the "internal state" of the corpus object, not the corpus data!

        To save the corpus data, use the `serialize` method of your desired output format
        instead, e.g. :meth:`gensim.corpora.mmcorpus.MmCorpus.serialize`.

        """
        import warnings
        warnings.warn(
            "corpus.save() stores only the (tiny) iteration object in memory; "
            "to serialize the actual corpus content, use e.g. MmCorpus.serialize(corpus)"
        )
        super(CorpusABC, self).save(*args, **kwargs)

    def __len__(self):
        """Get the corpus size = the total number of documents in it."""
        raise NotImplementedError("must override __len__() before calling len(corpus)")

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save `corpus` to disk.

        Some formats support saving the dictionary (`feature_id -> word` mapping),
        which can be provided by the optional `id2word` parameter.

        Notes
        -----
        Some corpora also support random access via document indexing, so that the documents on disk
        can be accessed in O(1) time (see the :class:`gensim.corpora.indexedcorpus.IndexedCorpus` base class).

        In this case, :meth:`~gensim.interfaces.CorpusABC.save_corpus` is automatically called internally by
        :func:`serialize`, which does :meth:`~gensim.interfaces.CorpusABC.save_corpus` plus saves the index
        at the same time.

        Calling :func:`serialize() is preferred to calling :meth:`gensim.interfaces.CorpusABC.save_corpus`.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        id2word : :class:`~gensim.corpora.Dictionary`, optional
            Dictionary of corpus.
        metadata : bool, optional
            Write additional metadata to a separate too?

        """
        raise NotImplementedError('cannot instantiate abstract base class')


class TransformedCorpus(CorpusABC):
    """Interface for corpora that are the result of an online (streamed) transformation."""
    def __init__(self, obj, corpus, chunksize=None, **kwargs):
        """

        Parameters
        ----------
        obj : object
            A transformation :class:`~gensim.interfaces.TransformationABC` object that will be applied
            to each document from `corpus` during iteration.
        corpus : iterable of list of (int, number)
            Corpus in bag-of-words format.
        chunksize : int, optional
            If provided, a slightly more effective processing will be performed by grouping documents from `corpus`.

        """
        self.obj, self.corpus, self.chunksize = obj, corpus, chunksize
        # add the new parameters like per_word_topics to base class object of LdaModel
        for key, value in kwargs.items():
            setattr(self.obj, key, value)
        self.metadata = False

    def __len__(self):
        """Get corpus size."""
        return len(self.corpus)

    def __iter__(self):
        """Iterate over the corpus, applying the selected transformation.

        If `chunksize` was set in the constructor, works in "batch-manner" (more efficient).

        Yields
        ------
        list of (int, number)
            Documents in the sparse Gensim bag-of-words format.

        """
        if self.chunksize:
            for chunk in utils.grouper(self.corpus, self.chunksize):
                for transformed in self.obj.__getitem__(chunk, chunksize=None):
                    yield transformed
        else:
            for doc in self.corpus:
                yield self.obj[doc]

    def __getitem__(self, docno):
        """Transform the document at position `docno` within `corpus` specified in the constructor.

        Parameters
        ----------
        docno : int
            Position of the document to transform. Document offset inside `self.corpus`.

        Notes
        -----
        `self.corpus` must support random indexing.

        Returns
        -------
        list of (int, number)
            Transformed document in the sparse Gensim bag-of-words format.

        Raises
        ------
        RuntimeError
            If corpus doesn't support index slicing (`__getitem__` doesn't exists).

        """
        if hasattr(self.corpus, '__getitem__'):
            return self.obj[self.corpus[docno]]
        else:
            raise RuntimeError('Type {} does not support slicing.'.format(type(self.corpus)))


class TransformationABC(utils.SaveLoad):

    def __getitem__(self, vec):
        raise NotImplementedError('cannot instantiate abstract base class')

    def _apply(self, corpus, chunksize=None, **kwargs):
        return TransformedCorpus(self, corpus, chunksize, **kwargs)


class SimilarityABC(utils.SaveLoad):
    def __init__(self, corpus):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def get_similarities(self, doc):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def __getitem__(self, query):
        is_corpus, query = utils.is_corpus(query)
        if self.normalize:
            # self.normalize only works if the input is a plain gensim vector/corpus (as
            # advertised in the doc). in fact, input can be a numpy or scipy.sparse matrix
            # as well, but in that case assume tricks are happening and don't normalize
            # anything (self.normalize has no effect).
            if not matutils.ismatrix(query):
                if is_corpus:
                    query = [matutils.unitvec(v) for v in query]
                else:
                    query = matutils.unitvec(query)
        result = self.get_similarities(query)

        if self.num_best is None:
            return result

        # if maintain_sparsity is True, result is scipy sparse. Sort, clip the
        # topn and return as a scipy sparse matrix.
        if getattr(self, 'maintain_sparsity', False):
            return matutils.scipy2scipy_clipped(result, self.num_best)

        # if the input query was a corpus (=more documents), compute the top-n
        # most similar for each document in turn
        if matutils.ismatrix(result):
            return [matutils.full2sparse_clipped(v, self.num_best) for v in result]
        else:
            # otherwise, return top-n of the single input document
            return matutils.full2sparse_clipped(result, self.num_best)

    def __iter__(self):
        # turn off query normalization (vectors in the index are assumed to be already normalized)
        norm = self.normalize
        self.normalize = False

        # Try to compute similarities in bigger chunks of documents (not
        # one query = a single document after another). The point is, a
        # bigger query of N documents is faster than N small queries of one
        # document.
        #
        # After computing similarities of the bigger query in `self[chunk]`,
        # yield the resulting similarities one after another, so that it looks
        # exactly the same as if they had been computed with many small queries.
        try:
            chunking = self.chunksize > 1
        except AttributeError:
            # chunking not supported; fall back to the (slower) mode of 1 query=1 document
            chunking = False
        if chunking:
            # assumes `self.corpus` holds the index as a 2-d numpy array.
            # this is true for MatrixSimilarity and SparseMatrixSimilarity, but
            # may not be true for other (future) classes..?
            for chunk_start in range(0, self.index.shape[0], self.chunksize):
                # scipy.sparse doesn't allow slicing beyond real size of the matrix
                # (unlike numpy). so, clip the end of the chunk explicitly to make
                # scipy.sparse happy
                chunk_end = min(self.index.shape[0], chunk_start + self.chunksize)
                chunk = self.index[chunk_start: chunk_end]
                for sim in self[chunk]:
                    yield sim
        else:
            for doc in self.index:
                yield self[doc]

        # restore old normalization value
        self.normalize = norm

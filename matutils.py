from __future__ import with_statement
import logging
import math
import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.special_matrices import triu
from scipy.special import psi


logger = logging.getLogger(__name__)


def blas(name, ndarray):
    return scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]


def argsort(x, topn=None, reverse=False):
    x = np.asarray(x)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))


def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
    try:
        if num_terms is None:
            num_terms = corpus.num_terms
        if num_docs is None:
            num_docs = corpus.num_docs
        if num_nnz is None:
            num_nnz = corpus.num_nnz
    except AttributeError:
        pass
    if printprogress:
        logger.info("creating sparse matrix from corpus")
    if num_terms is not None and num_docs is not None and num_nnz is not None:
        posnow, indptr = 0, [0]
        indices = np.empty((num_nnz,), dtype=np.int32)
        data = np.empty((num_nnz,), dtype=dtype)
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i/%i", docno, num_docs)
            posnext = posnow + len(doc)
            indices[posnow: posnext], data[posnow: posnext] = zip(*doc) if doc else ([], [])
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, "mismatch between supplied and computed number of non-zeros"
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        num_nnz, data, indices, indptr = 0, [], [], [0]
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i", docno)
            doc_indices, doc_data = zip(*doc) if doc else ([], [])
            indices.extend(doc_indices)
            data.extend(doc_data)
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        data = np.asarray(data, dtype=dtype)
        indices = np.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result


def pad(mat, padrow, padcol):
    if padrow < 0:
        padrow = 0
    if padcol < 0:
        padcol = 0
    rows, cols = mat.shape
    return np.block([
        [mat, np.zeros((rows, padcol))],
        [np.zeros((padrow, cols + padcol))],
    ])


def zeros_aligned(shape, dtype, order='C', align=128):
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)
    start_index = -buffer.ctypes.data % align
    return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)


def ismatrix(m):
    return isinstance(m, np.ndarray) and m.ndim == 2 or scipy.sparse.issparse(m)


def any2sparse(vec, eps=1e-9):
    if isinstance(vec, np.ndarray):
        return dense2vec(vec, eps)
    if scipy.sparse.issparse(vec):
        return scipy2sparse(vec, eps)
    return [(int(fid), float(fw)) for fid, fw in vec if np.abs(fw) > eps]


def scipy2scipy_clipped(matrix, topn, eps=1e-9):
    if not scipy.sparse.issparse(matrix):
        raise ValueError("'%s' is not a scipy sparse vector." % matrix)
    if topn <= 0:
        return scipy.sparse.csr_matrix([])
    if matrix.shape[0] == 1:
        biggest = argsort(abs(matrix.data), topn, reverse=True)
        indices, data = matrix.indices.take(biggest), matrix.data.take(biggest)
        return scipy.sparse.csr_matrix((data, indices, [0, len(indices)]))
    else:
        matrix_indices = []
        matrix_data = []
        matrix_indptr = [0]
        matrix_abs = abs(matrix)
        for i in range(matrix.shape[0]):
            v = matrix.getrow(i)
            v_abs = matrix_abs.getrow(i)
            biggest = argsort(v_abs.data, topn, reverse=True)
            indices, data = v.indices.take(biggest), v.data.take(biggest)
            matrix_data.append(data)
            matrix_indices.append(indices)
            matrix_indptr.append(matrix_indptr[-1] + min(len(indices), topn))
        matrix_indices = np.concatenate(matrix_indices).ravel()
        matrix_data = np.concatenate(matrix_data).ravel()
        return scipy.sparse.csr.csr_matrix(
            (matrix_data, matrix_indices, matrix_indptr),
            shape=(matrix.shape[0], np.max(matrix_indices) + 1)
        )


def scipy2sparse(vec, eps=1e-9):
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return [(int(pos), float(val)) for pos, val in zip(vec.indices, vec.data) if np.abs(val) > eps]


class Scipy2Corpus:
    def __init__(self, vecs):
        self.vecs = vecs

    def __iter__(self):
        for vec in self.vecs:
            if isinstance(vec, np.ndarray):
                yield full2sparse(vec)
            else:
                yield scipy2sparse(vec)

    def __len__(self):
        return len(self.vecs)


def sparse2full(doc, length):
    result = np.zeros(length, dtype=np.float32)
    doc = ((int(id_), float(val_)) for (id_, val_) in doc)
    doc = dict(doc)
    result[list(doc)] = list(doc.values())
    return result


def full2sparse(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))


dense2vec = full2sparse


def full2sparse_clipped(vec, topn, eps=1e-9):
    if topn <= 0:
        return []
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    biggest = nnz.take(argsort(abs(vec).take(nnz), topn, reverse=True))
    return list(zip(biggest, vec.take(biggest)))


def corpus2dense(corpus, num_terms, num_docs=None, dtype=np.float32):
    if num_docs is not None:
        docno, result = -1, np.empty((num_terms, num_docs), dtype=dtype)
        for docno, doc in enumerate(corpus):
            result[:, docno] = sparse2full(doc, num_terms)
        assert docno + 1 == num_docs
    else:
        result = np.column_stack([sparse2full(doc, num_terms) for doc in corpus])
    return result.astype(dtype)


class Dense2Corpus:
    def __init__(self, dense, documents_columns=True):
        if documents_columns:
            self.dense = dense.T
        else:
            self.dense = dense

    def __iter__(self):
        for doc in self.dense:
            yield full2sparse(doc.flat)

    def __len__(self):
        return len(self.dense)


class Sparse2Corpus:
    def __init__(self, sparse, documents_columns=True):
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T

    def __iter__(self):
        for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        return self.sparse.shape[1]

    def __getitem__(self, key):
        sparse = self.sparse
        if isinstance(key, int):
            iprev = self.sparse.indptr[key]
            inow = self.sparse.indptr[key + 1]
            return list(zip(sparse.indices[iprev:inow], sparse.data[iprev:inow]))

        sparse = self.sparse.__getitem__((slice(None, None, None), key))
        return Sparse2Corpus(sparse)


def veclen(vec):
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
    assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    return length


def ret_normalized_vec(vec, length):
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)


def ret_log_normalize_vec(vec, axis=1):
    log_max = 100.0
    if len(vec.shape) == 1:
        max_val = np.max(vec)
        log_shift = log_max - np.log(len(vec) + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift))
        log_norm = np.log(tot) - log_shift
        vec -= log_norm
    else:
        if axis == 1:
            max_val = np.max(vec, 1)
            log_shift = log_max - np.log(vec.shape[1] + 1.0) - max_val
            tot = np.sum(np.exp(vec + log_shift[:, np.newaxis]), 1)
            log_norm = np.log(tot) - log_shift
            vec = vec - log_norm[:, np.newaxis]
        elif axis == 0:
            k = ret_log_normalize_vec(vec.T)
            return k[0].T, k[1]
        else:
            raise ValueError("'%s' is not a supported axis" % axis)
    return vec, log_norm


blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))


def unitvec(vec, norm='l2', return_norm=False):
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if norm == 'unique':
            veclen = vec.nnz
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            vec /= veclen
            if return_norm:
                return vec, veclen
            else:
                return vec
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    try:
        first = next(iter(vec))
    except StopIteration:
        if return_norm:
            return vec, 1.0
        else:
            return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        if norm == 'unique':
            length = 1.0 * len(vec)
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        if return_norm:
            return ret_normalized_vec(vec, length), length
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")


def cossim(vec1, vec2):
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum(val * val for val in vec1.values()))
    vec2len = 1.0 * math.sqrt(sum(val * val for val in vec2.values()))
    assert vec1len > 0.0 and vec2len > 0.0, "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1
    result = sum(value * vec2.get(index, 0.0) for index, value in vec1.items())
    result /= vec1len * vec2len
    return result


def isbow(vec):
    if scipy.sparse.issparse(vec):
        vec = vec.todense().tolist()
    try:
        id_, val_ = vec[0]
        int(id_), float(val_)
    except IndexError:
        return True
    except (ValueError, TypeError):
        return False
    return True


def _convert_vec(vec1, vec2, num_features=None):
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        if num_features is not None:
            dense1 = sparse2full(vec1, num_features)
            dense2 = sparse2full(vec2, num_features)
            return dense1, dense2
        else:
            max_len = max(len(vec1), len(vec2))
            dense1 = sparse2full(vec1, max_len)
            dense2 = sparse2full(vec2, max_len)
            return dense1, dense2
    else:
        if len(vec1) == 1:
            vec1 = vec1[0]
        if len(vec2) == 1:
            vec2 = vec2[0]
        return vec1, vec2


def kullback_leibler(vec1, vec2, num_features=None):
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    return entropy(vec1, vec2)


def jensen_shannon(vec1, vec2, num_features=None):
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))


def hellinger(vec1, vec2):
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        vec1, vec2 = dict(vec1), dict(vec2)
        indices = set(list(vec1.keys()) + list(vec2.keys()))
        sim = np.sqrt(
            0.5 * sum((np.sqrt(vec1.get(index, 0.0)) - np.sqrt(vec2.get(index, 0.0)))**2 for index in indices)
        )
        return sim
    else:
        sim = np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2))**2).sum())
        return sim


def jaccard(vec1, vec2):
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        union = sum(weight for id_, weight in vec1) + sum(weight for id_, weight in vec2)
        vec1, vec2 = dict(vec1), dict(vec2)
        intersection = 0.0
        for feature_id, feature_weight in vec1.items():
            intersection += min(feature_weight, vec2.get(feature_id, 0.0))
        return 1 - float(intersection) / float(union)
    else:
        if isinstance(vec1, np.ndarray):
            vec1 = vec1.tolist()
        if isinstance(vec2, np.ndarray):
            vec2 = vec2.tolist()
        vec1 = set(vec1)
        vec2 = set(vec2)
        intersection = vec1 & vec2
        union = vec1 | vec2
        return 1 - float(len(intersection)) / float(len(union))


def jaccard_distance(set1, set2):
    union_cardinality = len(set1 | set2)
    if union_cardinality == 0:
        return 1.

    return 1. - float(len(set1 & set2)) / float(union_cardinality)


try:
    from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation

except ImportError:
    def logsumexp(x):
        x_max = np.max(x)
        x = np.log(np.sum(np.exp(x - x_max)))
        x += x_max

        return x

    def mean_absolute_difference(a, b):
        return np.mean(np.abs(a - b))

    def dirichlet_expectation(alpha):
        if len(alpha.shape) == 1:
            result = psi(alpha) - psi(np.sum(alpha))
        else:
            result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        return result.astype(alpha.dtype, copy=False)


def qr_destroy(la):
    a = np.asfortranarray(la[0])
    del la[0], la
    m, n = a.shape
    logger.debug("computing QR of %s dense matrix", str(a.shape))
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n:
        qr = qr[:, :m]
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    q, work, info = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, "qr failed"
    assert q.flags.f_contiguous
    return q, r


class MmWriter:
    HEADER_LINE = b'%%MatrixMarket matrix coordinate real general\n'

    def __init__(self, fname):
        self.fname = fname
        if fname.endswith(".gz") or fname.endswith('.bz2'):
            raise NotImplementedError("compressed output not supported with MmWriter")
        self.fout = utils.open(self.fname, 'wb+')
        self.headers_written = False

    def write_headers(self, num_docs, num_terms, num_nnz):
        self.fout.write(MmWriter.HEADER_LINE)

        if num_nnz < 0:
            logger.info("saving sparse matrix to %s", self.fname)
            self.fout.write(utils.to_utf8(' ' * 50 + '\n'))
        else:
            logger.info(
                "saving sparse %sx%s matrix with %i non-zero entries to %s",
                num_docs, num_terms, num_nnz, self.fname
            )
            self.fout.write(utils.to_utf8('%s %s %s\n' % (num_docs, num_terms, num_nnz)))
        self.last_docno = -1
        self.headers_written = True

    def fake_headers(self, num_docs, num_terms, num_nnz):
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(utils.to_utf8(stats))

    def write_vector(self, docno, vector):
        assert self.headers_written, "must write Matrix Market file headers before writing data!"
        assert self.last_docno < docno, "documents %i and %i not in sequential order!" % (self.last_docno, docno)
        vector = sorted((i, w) for i, w in vector if abs(w) > 1e-12)
        for termid, weight in vector:
            self.fout.write(utils.to_utf8("%i %i %s\n" % (docno + 1, termid + 1, weight)))
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False, num_terms=None, metadata=False):
        mw = MmWriter(fname)
        mw.write_headers(-1, -1, -1)
        _num_terms, num_nnz = 0, 0
        docno, poslast = -1, -1
        offsets = []
        if hasattr(corpus, 'metadata'):
            orig_metadata = corpus.metadata
            corpus.metadata = metadata
            if metadata:
                docno2metadata = {}
        else:
            metadata = False
        for docno, doc in enumerate(corpus):
            if metadata:
                bow, data = doc
                docno2metadata[docno] = data
            else:
                bow = doc
            if docno % progress_cnt == 0:
                logger.info("PROGRESS: saving document #%i", docno)
            if index:
                posnow = mw.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow
            max_id, veclen = mw.write_vector(docno, bow)
            _num_terms = max(_num_terms, 1 + max_id)
            num_nnz += veclen
        if metadata:
            utils.pickle(docno2metadata, fname + '.metadata.cpickle')
            corpus.metadata = orig_metadata

        num_docs = docno + 1
        num_terms = num_terms or _num_terms

        if num_docs * num_terms != 0:
            logger.info(
                "saved %ix%i matrix, density=%.3f%% (%i/%i)",
                num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms
            )
        mw.fake_headers(num_docs, num_terms, num_nnz)

        mw.close()
        if index:
            return offsets

    def __del__(self):
        self.close()

    def close(self):
        logger.debug("closing %s", self.fname)
        if hasattr(self, 'fout'):
            self.fout.close()


try:
    from _mmreader import MmReader
except ImportError:
    raise utils.NO_CYTHON

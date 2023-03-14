from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types

import numpy as np
import scipy.sparse
from smart_open import open

gensim_version = '4.2.0'
logger = logging.getLogger(__name__)

PICKLE_PROTOCOL = 4

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

NO_CYTHON = RuntimeError(
    "Compiled extensions are unavailable. "
    "If you've installed from a package, ask the package maintainer to include compiled extensions. "
    "If you're building Gensim from source yourself, install Cython and a C compiler, and then "
    "run `python setup.py build_ext --inplace` to retry. "
)

default_prng = np.random.default_rng()


def get_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a np.random.RandomState instance' % seed)


def synchronous(tlockname):
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = getattr(self, tlockname)
            logger.debug("acquiring lock %r for %s", tlockname, func.__name__)

            with tlock:
                logger.debug("acquired lock %r for %s", tlockname, func.__name__)
                result = func(self, *args, **kwargs)
                logger.debug("releasing lock %r for %s", tlockname, func.__name__)
                return result
        return _synchronizer
    return _synched


def file_or_filename(input):
    if isinstance(input, str):
        return open(input, 'rb')
    else:
        input.seek(0)
        return input


@contextmanager
def open_file(input):
    mgr = file_or_filename(input)
    exc = False
    try:
        yield mgr
    except Exception:
        exc = True
        if not isinstance(input, str) or not mgr.__exit__(*sys.exc_info()):
            raise
    finally:
        if not exc and isinstance(input, str):
            mgr.__exit__(None, None, None)


def deaccent(text):
    if not isinstance(text, str):
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def copytree_hardlink(source, dest):
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2


def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def any2utf8(text, errors='strict', encoding='utf8'):
    if isinstance(text, str):
        return text.encode('utf8')
    return str(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


to_unicode = any2unicode


def call_on_class_only(*args, **kwargs):
    raise AttributeError('This method should be called on a class object.')


class SaveLoad:
    def add_lifecycle_event(self, event_name, log_level=logging.INFO, **event):
        event_dict = deepcopy(event)
        event_dict['datetime'] = datetime.now().isoformat()
        event_dict['gensim'] = gensim_version
        event_dict['python'] = sys.version
        event_dict['platform'] = platform.platform()
        event_dict['event'] = event_name
        if not hasattr(self, 'lifecycle_events'):
            logger.debug("starting a new internal lifecycle event log for %s", self.__class__.__name__)
            self.lifecycle_events = []
        if log_level:
            logger.log(log_level, "%s lifecycle event %s", self.__class__.__name__, event_dict)

        if self.lifecycle_events is not None:
            self.lifecycle_events.append(event_dict)

    @classmethod
    def load(cls, fname, mmap=None):
        logger.info("loading %s object from %s", cls.__name__, fname)
        compress, subname = SaveLoad._adapt_by_suffix(fname)
        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        obj.add_lifecycle_event("loaded", fname=fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        def mmap_error(obj, filename):
            return IOError(
                'Cannot mmap compressed object %s in file %s. ' % (obj, filename)
                + 'Use `load(fname, mmap=None)` or uncompress files manually.'
            )
        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s", attrib, cfname, mmap)
            with ignore_deprecation_warning():
                getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            with ignore_deprecation_warning():
                setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            with ignore_deprecation_warning():
                setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None", attrib)
            with ignore_deprecation_warning():
                setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
        return compress, lambda *args: '.'.join(args + (suffix,))

    def _smart_save(
            self, fname,separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=PICKLE_PROTOCOL):
        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(
            fname, separately, sep_limit, ignore, pickle_protocol, compress, subname,
        )
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            for obj, asides in restores:
                for attrib, val in asides.items():
                    with ignore_deprecation_warning():
                        setattr(obj, attrib, val)
        logger.info("saved %s", fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in self.__dict__.items():
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)
        with ignore_deprecation_warning():
            for attrib in separately + list(ignore):
                if hasattr(self, attrib):
                    asides[attrib] = getattr(self, attrib)
                    delattr(self, attrib)
        recursive_saveloads = []
        restores = []
        for attrib, val in self.__dict__.items():
            if hasattr(val, '_save_specials'):
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore, pickle_protocol, compress, subname))
        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in asides.items():
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s", attrib, subname(fname, attrib))
                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))
                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(
                            subname(fname, attrib, 'sparse'),
                            data=val.data,
                            indptr=val.indptr,
                            indices=val.indices
                        )
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s", attrib)
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except Exception:
            for attrib, val in asides.items():
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(
            self, fname_or_handle,
            separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=PICKLE_PROTOCOL,
        ):
        self.add_lifecycle_event(
            "saving",
            fname_or_handle=str(fname_or_handle),
            separately=str(separately),
            sep_limit=sep_limit,
            ignore=ignore,
        )
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object", self.__class__.__name__)
        except TypeError:
            self._smart_save(fname_or_handle, separately, sep_limit, ignore, pickle_protocol=pickle_protocol)


def identity(p):
    return p


def get_max_id(corpus):
    maxid = -1
    for document in corpus:
        if document:
            maxid = max(maxid, max(fieldid for fieldid, _ in document))
    return maxid


class FakeDict:
    def __init__(self, num_terms):
        self.num_terms = num_terms

    def __str__(self):
        return "%s<num_terms=%s>" % (self.__class__.__name__, self.num_terms)

    def __getitem__(self, val):
        if 0 <= val < self.num_terms:
            return str(val)
        raise ValueError("internal id out of bounds (%s, expected <0..%s))" % (val, self.num_terms))

    def __contains__(self, val):
        return 0 <= val < self.num_terms

    def iteritems(self):
        for i in range(self.num_terms):
            yield i, str(i)

    def keys(self):
        return [self.num_terms - 1]

    def __len__(self):
        return self.num_terms

    def get(self, val, default=None):
        if 0 <= val < self.num_terms:
            return str(val)
        return default


def dict_from_corpus(corpus):
    num_terms = 1 + get_max_id(corpus)
    id2word = FakeDict(num_terms)
    return id2word


def is_corpus(obj):
    try:
        if 'Corpus' in obj.__class__.__name__:
            return True, obj
    except Exception:
        pass
    try:
        if hasattr(obj, 'next') or hasattr(obj, '__next__'):
            doc1 = next(obj)
            obj = itertools.chain([doc1], obj)
        else:
            doc1 = next(iter(obj))
        if len(doc1) == 0:
            return True, obj
        id1, val1 = next(iter(doc1))
    except Exception:
        return False, obj
    return True, obj


def get_my_ip():
    import socket
    try:
        from Pyro4.naming import locateNS
        ns = locateNS()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ns._pyroUri.host, ns._pyroUri.port))
        result, port = s.getsockname()
    except Exception:
        try:
            import commands
            result = commands.getoutput("ifconfig").split("\n")[1].split()[1][5:]
            if len(result.split('.')) != 4:
                raise Exception()
        except Exception:
            result = socket.gethostbyname(socket.gethostname())
    return result


class RepeatCorpus(SaveLoad):
    def __init__(self, corpus, reps):
        self.corpus = corpus
        self.reps = reps

    def __iter__(self):
        return itertools.islice(itertools.cycle(self.corpus), self.reps)


class RepeatCorpusNTimes(SaveLoad):
    def __init__(self, corpus, n):
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document


class ClippedCorpus(SaveLoad):
    def __init__(self, corpus, max_docs=None):
        self.corpus = corpus
        self.max_docs = max_docs

    def __iter__(self):
        return itertools.islice(self.corpus, self.max_docs)

    def __len__(self):
        return min(self.max_docs, len(self.corpus))


class SlicedCorpus(SaveLoad):
    def __init__(self, corpus, slice_):
        self.corpus = corpus
        self.slice_ = slice_
        self.length = None

    def __iter__(self):
        if hasattr(self.corpus, 'index') and len(self.corpus.index) > 0:
            return (self.corpus.docbyoffset(i) for i in self.corpus.index[self.slice_])
        return itertools.islice(self.corpus, self.slice_.start, self.slice_.stop, self.slice_.step)

    def __len__(self):
        if self.length is None:
            if isinstance(self.slice_, (list, np.ndarray)):
                self.length = len(self.slice_)
            elif isinstance(self.slice_, slice):
                (start, end, step) = self.slice_.indices(len(self.corpus.index))
                diff = end - start
                self.length = diff // step + (diff % step > 0)
            else:
                self.length = sum(1 for x in self)

        return self.length


def safe_unichr(intval):
    try:
        return chr(intval)
    except ValueError:
        s = "\\U%08x" % intval
        return s.decode('unicode-escape')


def decode_htmlentities(text):
    def substitute_entity(match):
        try:
            ent = match.group(3)
            if match.group(1) == "#":
                if match.group(2) == '':
                    return safe_unichr(int(ent))
                elif match.group(2) in ['x', 'X']:
                    return safe_unichr(int(ent, 16))
            else:
                cp = n2cp.get(ent)
                if cp:
                    return safe_unichr(cp)
                else:
                    return match.group()
        except Exception:
            return match.group()
    return RE_HTML_ENTITY.sub(substitute_entity, text)


def chunkize_serial(iterable, chunksize, as_numpy=False, dtype=np.float32):
    it = iter(iterable)
    while True:
        if as_numpy:
            wrapped_chunk = [[np.array(doc, dtype=dtype) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        yield wrapped_chunk.pop()


grouper = chunkize_serial


class InputQueue(multiprocessing.Process):
    def __init__(self, q, corpus, chunksize, maxsize, as_numpy):
        super(InputQueue, self).__init__()
        self.q = q
        self.maxsize = maxsize
        self.corpus = corpus
        self.chunksize = chunksize
        self.as_numpy = as_numpy

    def run(self):
        it = iter(self.corpus)
        while True:
            chunk = itertools.islice(it, self.chunksize)
            if self.as_numpy:
                wrapped_chunk = [[np.asarray(doc) for doc in chunk]]
            else:
                wrapped_chunk = [list(chunk)]

            if not wrapped_chunk[0]:
                self.q.put(None, block=True)
                break

            try:
                qsize = self.q.qsize()
            except NotImplementedError:
                qsize = '?'
            logger.debug("prepared another chunk of %i documents (qsize=%s)", len(wrapped_chunk[0]), qsize)
            self.q.put(wrapped_chunk.pop(), block=True)


if os.name == 'nt' or (sys.platform == "darwin" and sys.version_info >= (3, 8)):
    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        if maxsize > 0:
            entity = "Windows" if os.name == 'nt' else "OSX with python3.8+"
            warnings.warn("detected %s; aliasing chunkize to chunkize_serial" % entity)
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk
else:
    def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
        assert chunksize > 0
        if maxsize > 0:
            q = multiprocessing.Queue(maxsize=maxsize)
            worker = InputQueue(q, corpus, chunksize, maxsize=maxsize, as_numpy=as_numpy)
            worker.daemon = True
            worker.start()
            while True:
                chunk = [q.get(block=True)]
                if chunk[0] is None:
                    break
                yield chunk.pop()
        else:
            for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
                yield chunk


def smart_extension(fname, ext):
    fname, oext = os.path.splitext(fname)
    if oext.endswith('.bz2'):
        fname = fname + oext[:-4] + ext + '.bz2'
    elif oext.endswith('.gz'):
        fname = fname + oext[:-3] + ext + '.gz'
    else:
        fname = fname + oext + ext
    return fname


def pickle(obj, fname, protocol=PICKLE_PROTOCOL):
    with open(fname, 'wb') as fout:
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    with open(fname, 'rb') as f:
        return _pickle.load(f, encoding='latin1')


def revdict(d):
    return {v: k for (k, v) in dict(d).items()}


def deprecated(reason):
    if isinstance(reason, str):
        def decorator(func):
            fmt = "Call to deprecated `{name}` ({reason})."

            @wraps(func)
            def new_func1(*args, **kwargs):
                warnings.warn(
                    fmt.format(name=func.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)

            return new_func1
        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason
        fmt = "Call to deprecated `{name}`."
        @wraps(func)
        def new_func2(*args, **kwargs):
            warnings.warn(
                fmt.format(name=func.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func2
    else:
        raise TypeError(repr(type(reason)))


@contextmanager
def ignore_deprecation_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


@deprecated("Function will be removed in 4.0.0")
def toptexts(query, texts, index, n=10):
    sims = index[query]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    return [(topid, topcosine, texts[topid]) for topid, topcosine in sims[:n]]


def randfname(prefix='gensim'):
    randpart = hex(random.randint(0, 0xffffff))[2:]
    return os.path.join(tempfile.gettempdir(), prefix + randpart)


@deprecated("Function will be removed in 4.0.0")
def upload_chunked(server, docs, chunksize=1000, preprocess=None):
    start = 0
    for chunk in grouper(docs, chunksize):
        end = start + len(chunk)
        logger.info("uploading documents %i-%i", start, end - 1)
        if preprocess is not None:
            pchunk = []
            for doc in chunk:
                doc['tokens'] = preprocess(doc['text'])
                del doc['text']
                pchunk.append(doc)
            chunk = pchunk
        server.buffer(chunk)
        start = end


def getNS(host=None, port=None, broadcast=True, hmac_key=None):
    import Pyro4
    try:
        return Pyro4.locateNS(host, port, broadcast, hmac_key)
    except Pyro4.errors.NamingError:
        raise RuntimeError("Pyro name server not found")


def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None, ns_conf=None):
    if ns_conf is None:
        ns_conf = {}
    if random_suffix:
        name += '.' + hex(random.randint(0, 0xffffff))[2:]

    import Pyro4
    with getNS(**ns_conf) as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')", name, uri)
            daemon.requestLoop()


def mock_data_row(dim=1000, prob_nnz=0.5, lam=1.0):
    nnz = np.random.uniform(size=(dim,))
    return [(i, float(np.random.poisson(lam=lam) + 1.0)) for i in range(dim) if nnz[i] < prob_nnz]


def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    return [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam) for _ in range(n_items)]


def prune_vocab(vocab, min_reduce, trim_rule=None):
    result = 0
    old_len = len(vocab)
    for w in list(vocab):
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):
            result += vocab[w]
            del vocab[w]
    logger.info(
        "pruned out %i tokens with count <=%i (before %i, after %i)",
        old_len - len(vocab), min_reduce, old_len, len(vocab)
    )
    return result


def trim_vocab_by_freq(vocab, topk, trim_rule=None):
    if topk >= len(vocab):
        return

    min_count = heapq.nlargest(topk, vocab.values())[-1]
    prune_vocab(vocab, min_count, trim_rule=trim_rule)


def merge_counts(dict1, dict2):
    for word, freq in dict2.items():
        if word in dict1:
            dict1[word] += freq
        else:
            dict1[word] = freq

    return dict1


def qsize(queue):
    try:
        return queue.qsize()
    except NotImplementedError:
        return -1


RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2


def keep_vocab_item(word, count, min_count, trim_rule=None):
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res


def check_output(stdout=subprocess.PIPE, *popenargs, **kwargs):
    try:
        logger.debug("COMMAND: %s %s", popenargs, kwargs)
        process = subprocess.Popen(stdout=stdout, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output
    except KeyboardInterrupt:
        process.terminate()
        raise


def sample_dict(d, n=10, use_random=True):
    selected_keys = random.sample(list(d), min(len(d), n)) if use_random else itertools.islice(d.keys(), n)
    return [(key, d[key]) for key in selected_keys]


def strided_windows(ndarray, window_size):
    ndarray = np.asarray(ndarray)
    if window_size == ndarray.shape[0]:
        return np.array([ndarray])
    elif window_size > ndarray.shape[0]:
        return np.ndarray((0, 0))

    stride = ndarray.strides[0]
    return np.lib.stride_tricks.as_strided(
        ndarray, shape=(ndarray.shape[0] - window_size + 1, window_size),
        strides=(stride, stride))


def iter_windows(texts, window_size, copy=False, ignore_below_size=True, include_doc_num=False):
    for doc_num, document in enumerate(texts):
        for window in _iter_windows(document, window_size, copy, ignore_below_size):
            if include_doc_num:
                yield (doc_num, window)
            else:
                yield window


def _iter_windows(document, window_size, copy=False, ignore_below_size=True):
    doc_windows = strided_windows(document, window_size)
    if doc_windows.shape[0] == 0:
        if not ignore_below_size:
            yield document.copy() if copy else document
    else:
        for doc_window in doc_windows:
            yield doc_window.copy() if copy else doc_window


def flatten(nested_list):
    return list(lazy_flatten(nested_list))


def lazy_flatten(nested_list):
    for el in nested_list:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def save_as_line_sentence(corpus, filename):
    with open(filename, mode='wb', encoding='utf8') as fout:
        for sentence in corpus:
            line = any2unicode(' '.join(sentence) + '\n')
            fout.write(line)


def effective_n_jobs(n_jobs):
    if n_jobs == 0:
        raise ValueError('n_jobs == 0 in Parallel has no meaning')
    elif n_jobs is None:
        return 1
    elif n_jobs < 0:
        n_jobs = max(multiprocessing.cpu_count() + 1 + n_jobs, 1)
    return n_jobs


def is_empty(corpus):
    if scipy.sparse.issparse(corpus):
        return corpus.shape[1] == 0
    if isinstance(corpus, types.GeneratorType):
        return False
    try:
        first_doc = next(iter(corpus))
        return False
    except StopIteration:
        return True
    except Exception:
        return False

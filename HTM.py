import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi
from scipy.special import polygamma
import matutils
import interfaces
import utils
from dictionary import Dictionary
from matutils import(
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference,
)
import basemodel
from CoherenceModel import CoherenceModel
import Callback


logger = logging.getLogger(__name__)


def update_dir_prior(prior, N, logphat, rho):
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)
    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)
    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))
    dprior = -(gradf - b) / q
    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        logger.warning("updated prior is not positive")
    return prior


class LdaState(utils.SaveLoad):
    def __init__(self, eta, shape, dtype=np.float32):
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        return self.eta + self.sstats

    def get_Elogbeta(self):
        return dirichlet_expectation(self.get_lambda())

    @classmethod
    def load(cls, fname, *args, **kwargs):
        result = super(LdaState, cls).load(fname, *args, **kwargs)

        if not hasattr(result, 'dtype'):
            result.dtype = np.float64
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)

        return result


class HtmLdaBased(interfaces.TransformationABC, basemodel.BaseTopicModel):
    def __init__(self, preprocessed_data, corpus=None, num_topics=100, id2word=None, distributed=False, chunksize=2000,
                 passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None,
                 minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=np.float32):
        self.dtype = np.finfo(dtype).dtype
        docs = []
        v_all = []
        for doc in preprocessed_data:
            for d in doc:
                docs.append(d)
            v_all.append(Dictionary(doc))
        id2word = Dictionary(docs)
        c_all = []
        for i in range(0, len(preprocessed_data)):
            res = [v_all[i].doc2bow(words) for words in preprocessed_data[i]]
            c_all.append(res)

        corpus = []
        for corp in c_all:
            sub_corpus = []
            for c in corp:
                sub_corpus += c
            corpus.append(sub_corpus)
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality'
                             )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0
        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks
        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)
        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')
        assert self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms), (
            "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))
        self.random_state = utils.get_random_state(random_state)
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError("auto-optimizing alpha not implemented in distributed LDA")
            try:
                import Pyro4
                if ns_conf is None:
                    ns_conf = {}
                with utils.getNS(**ns_conf) as ns:
                    LDA_DISPATCHER_PREFIX = 'gensim.lda_dispatcher'
                    self.dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug("looking for dispatcher at %s" % str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(
                        id2word=self.id2word, num_topics=self.num_topics, chunksize=chunksize,
                        alpha=alpha, eta=eta, distributed=False
                    )
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info("using distributed version with %i workers", self.numworkers)
            except Exception as err:
                logger.error("failed to initialize distributed LDA (%s)", err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms), dtype=self.dtype)
        self.state.sstats[...] = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))
        assert self.eta.dtype == self.dtype
        assert self.expElogbeta.dtype == self.dtype
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            start = time.time()
            self.update(corpus, chunks_as_numpy=use_numpy)
            self.add_lifecycle_event("created",msg=f"trained {self} in {time.time() - start:.2f}s")

    def init_dir_prior(self, prior, name):
        if prior is None:
            prior = 'symmetric'
        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'eta'")
        is_auto = False
        if isinstance(prior, str):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0 / self.num_topics)
                init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)),
                                         dtype=self.dtype, count=prior_shape)
            elif prior == 'asymmetric':
                if name == 'eta':
                    raise ValueError("The 'asymmetric' option cannot be used for eta")
                init_prior = np.fromiter(
                    (1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape,
                )
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape)
                if name == 'alpha':
                    logger.info("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, (np.number, numbers.Real)):
            init_prior = np.fromiter((prior for i in range(prior_shape)), dtype=self.dtype)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)
        return init_prior, is_auto

    def __str__(self):
        return "%s<num_terms=%s, num_topics=%s, decay=%s, chunksize=%s>" % (self.__class__.__name__, self.num_terms,
                                                                            self.num_topics, self.decay, self.chunksize)

    def sync_state(self, current_Elogbeta=None):
        if current_Elogbeta is None:
            current_Elogbeta = self.state.get_Elogbeta()
        self.expElogbeta = np.exp(current_Elogbeta)
        assert self.expElogbeta.dtype == self.dtype

    def clear(self):
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        try:
            len(chunk)
        except TypeError:
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))
        gamma = self.random_state.gamma(100., 1. / 100., (len(chunk), self.num_topics)).astype(self.dtype, copy=False)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
        else:
            sstats = None
        converged = 0
        integer_types = (int, np.integer,)
        epsilon = np.finfo(self.dtype).eps
        for d, doc in enumerate(chunk):
            if len(doc) > 0 and not isinstance(doc[0][0], integer_types):
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.fromiter((cnt for _, cnt in doc), dtype=self.dtype, count=len(doc))
            gammad = gamma[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
            for _ in range(self.iterations):
                lastgamma = gammad
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
                meanchange = mean_absolute_difference(gammad, lastgamma)
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break
            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)
        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations", converged, len(chunk), self.iterations)
        if collect_sstats:
            sstats *= self.expElogbeta
            assert sstats.dtype == self.dtype
        assert gamma.dtype == self.dtype
        return gamma, sstats

    def do_estep(self, chunk, state=None):
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        state.numdocs += gamma.shape[0]
        assert gamma.dtype == self.dtype
        return gamma

    def update_alpha(self, gammat, rho):
        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N
        assert logphat.dtype == self.dtype
        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)
        logger.info("optimized alpha %s", list(self.alpha))
        assert self.alpha.dtype == self.dtype
        return self.alpha

    def update_eta(self, lambdat, rho):
        N = float(lambdat.shape[0])
        logphat = (sum(dirichlet_expectation(lambda_) for lambda_ in lambdat) / N).reshape((self.num_terms,))
        assert logphat.dtype == self.dtype
        self.eta = update_dir_prior(self.eta, N, logphat, rho)
        assert self.eta.dtype == self.dtype
        return self.eta

    def log_perplexity(self, chunk, total_docs=None):
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words
        )
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None, passes=None, update_every=None, eval_every=None,
               iterations=None, gamma_threshold=None, chunks_as_numpy=False):
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold
        try:
            lencorpus = len(corpus)
        except Exception:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return
        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)
        self.state.numdocs += lencorpus
        if update_every:
            updatetype = "online"
            if passes == 1:
                updatetype += " (single-pass)"
            else:
                updatetype += " (multi-pass)"
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)
        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s LDA training, %s topics, %i passes over "
            "the supplied corpus of %i documents, updating model once "
            "every %i documents, evaluating perplexity every %i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, passes, lencorpus,
            updateafter, evalafter, iterations,
            gamma_threshold
        )
        if updates_per_pass * passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )
        def rho():
            return pow(offset + pass_ + (self.num_updates / chunksize), -decay)
        if self.callbacks:
            callback = Callback(self.callbacks)
            callback.set_model(self)
            self.metrics = defaultdict(list)
        for pass_ in range(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
            dirty = False
            reallen = 0
            chunks = utils.grouper(corpus, chunksize, as_numpy=chunks_as_numpy, dtype=self.dtype)
            for chunk_no, chunk in enumerate(chunks):
                reallen += len(chunk)

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    self.log_perplexity(chunk, total_docs=lencorpus)

                if self.dispatcher:
                    logger.info(
                        "PROGRESS: pass %i, dispatching documents up to #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info(
                        "PROGRESS: pass %i, at document #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    gammat = self.do_estep(chunk, other)

                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())
                dirty = True
                del chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
                    dirty = False
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for metric, value in current_metrics.items():
                    self.metrics[metric].append(value)
            if dirty:
                if self.dispatcher:
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other
                dirty = False

    def do_mstep(self, rho, other, extra_pass=False):
        logger.debug("updating topics")
        previous_Elogbeta = self.state.get_Elogbeta()
        self.state.blend(rho, other)
        current_Elogbeta = self.state.get_Elogbeta()
        self.sync_state(current_Elogbeta)
        self.print_topics(5)
        diff = mean_absolute_difference(previous_Elogbeta.ravel(), current_Elogbeta.ravel())
        logger.info("topic diff=%f, rho=%f", diff, rho)
        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)
        if not extra_pass:
            self.num_updates += other.numdocs

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        for d, doc in enumerate(corpus):
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)
            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype
            score += sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))
        score *= subsample_ratio
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)
        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))
        return score

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            sort_alpha = self.alpha + 0.0001 * self.random_state.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]
        shown = []
        topic = self.state.get_lambda()
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()
            bestn = matutils.argsort(topic_, num_words, reverse=True)
            topic_ = [(self.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join('%.3f*"%s"' % (v, k) for k, v in topic_)
            shown.append((i, topic_))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic_)
        return shown

    def show_topic(self, topicid, topn=10):
        return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]

    def get_topics(self):
        topics = self.state.get_lambda()
        return topics / topics.sum(axis=1)[:, None]

    def get_topic_terms(self, topicid, topn=10):
        topic = self.get_topics()[topicid]
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def top_topics(self, corpus=None, texts=None, dictionary=None, window_size=None,
                   coherence='u_mass', topn=20, processes=-1):
        cm = CoherenceModel(
            model=self, corpus=corpus, texts=texts, dictionary=dictionary,
            window_size=window_size, coherence=coherence, topn=topn,
            processes=processes
        )
        coherence_scores = cm.get_coherence_per_topic()
        str_topics = []
        for topic in self.get_topics():
            bestn = matutils.argsort(topic, topn=topn, reverse=True)
            beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]
            str_topics.append(beststr)
        scored_topics = zip(str_topics, coherence_scores)
        return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)

        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        minimum_phi_value = max(minimum_phi_value, 1e-8)

        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            kwargs = dict(
                per_word_topics=per_word_topics,
                minimum_probability=minimum_probability,
                minimum_phi_value=minimum_phi_value
            )
            return self._apply(corpus, **kwargs)
        gamma, phis = self.inference([bow], collect_sstats=per_word_topics)
        topic_dist = gamma[0] / sum(gamma[0])
        document_topics = [
            (topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
            if topicvalue >= minimum_probability
        ]
        if not per_word_topics:
            return document_topics
        word_topic = []
        word_phi = []
        for word_type, weight in bow:
            phi_values = []
            phi_topic = []
            for topic_id in range(0, self.num_topics):
                if phis[topic_id][word_type] >= minimum_phi_value:
                    phi_values.append((phis[topic_id][word_type], topic_id))
                    phi_topic.append((topic_id, phis[topic_id][word_type]))
            word_phi.append((word_type, phi_topic))
            sorted_phi_values = sorted(phi_values, reverse=True)
            topics_sorted = [x[1] for x in sorted_phi_values]
            word_topic.append((word_type, topics_sorted))
        return document_topics, word_topic, word_phi

    def get_term_topics(self, word_id, minimum_probability=None):
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]
        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[topic_id][word_id] >= minimum_probability:
                values.append((topic_id, self.expElogbeta[topic_id][word_id]))
        return values

    def diff(self, other, distance="kullback_leibler", num_words=100,
             n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        distances = {
            "kullback_leibler": kullback_leibler,
            "hellinger": hellinger,
            "jaccard": jaccard_distance,
            "jensen_shannon": jensen_shannon
        }
        if distance not in distances:
            valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
            raise ValueError("Incorrect distance, valid only {}".format(valid_keys))
        if not isinstance(other, self.__class__):
            raise ValueError("The parameter `other` must be of type `{}`".format(self.__name__))
        distance_func = distances[distance]
        d1, d2 = self.get_topics(), other.get_topics()
        t1_size, t2_size = d1.shape[0], d2.shape[0]
        annotation_terms = None
        fst_topics = [{w for (w, _) in self.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        snd_topics = [{w for (w, _) in other.show_topic(topic, topn=num_words)} for topic in range(t2_size)]
        if distance == "jaccard":
            d1, d2 = fst_topics, snd_topics
        if diagonal:
            assert t1_size == t2_size, \
                "Both input models should have same no. of topics, " \
                "as the diagonal will only be valid in a square matrix"
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]
            z[topic] = distance_func(d1[topic1], d2[topic2])
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])
                pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]
                annotation_terms[topic] = [pos_tokens, neg_tokens]
        if normed:
            if np.abs(np.max(z)) > 1e-8:
                z /= np.max(z)
        return z, annotation_terms

    def __getitem__(self, bow, eps=None):
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs):
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)
        if 'id2word' not in ignore:
            utils.pickle(self.id2word, utils.smart_extension(fname, '.id2word'))
        if ignore is not None and ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            ignore = [e for e in ignore if e]
            ignore = list({'state', 'dispatcher', 'id2word'} | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']
        separately_explicit = ['expElogbeta', 'sstats']
        if (isinstance(self.alpha, str) and self.alpha == 'auto') or \
                (isinstance(self.alpha, np.ndarray) and len(self.alpha.shape) != 1):
            separately_explicit.append('alpha')
        if (isinstance(self.eta, str) and self.eta == 'auto') or \
                (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        if separately:
            if isinstance(separately, str):
                separately = [separately]
            separately = [e for e in separately if e]
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(HtmLdaBased, self).save(fname, ignore=ignore, separately=separately, *args, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(HtmLdaBased, cls).load(fname, *args, **kwargs)
        if not hasattr(result, 'random_state'):
            result.random_state = utils.get_random_state(None)
            logging.warning("random_state not set so using default value")
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)
        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = LdaState.load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)
        id2word_fname = utils.smart_extension(fname, '.id2word')
        if os.path.isfile(id2word_fname):
            try:
                result.id2word = utils.unpickle(id2word_fname)
            except Exception as e:
                logging.warning("failed to load id2word dictionary from %s: %s", id2word_fname, e)
        return result

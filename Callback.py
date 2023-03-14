import CoherenceModel
import logging
import copy
import numpy as np
from queue import Queue


try:
    from visdom import Visdom
    VISDOM_INSTALLED = True
except ImportError:
    VISDOM_INSTALLED = False


class Metric:
    def __str__(self):
        if self.title is not None:
            return self.title
        else:
            return type(self).__name__[:-6]

    def set_parameters(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def get_value(self):
        raise NotImplementedError("Please provide an implementation for `get_value` in your subclass.")


class CoherenceMetric(Metric):
    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None,
                 window_size=None, topn=10, logger=None, viz_env=None, title=None):
        self.corpus = corpus
        self.dictionary = dictionary
        self.coherence = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        self.model = None
        self.topics = None
        super(CoherenceMetric, self).set_parameters(**kwargs)
        cm = CoherenceModel.CoherenceModel(
            model=self.model, topics=self.topics, texts=self.texts, corpus=self.corpus,
            dictionary=self.dictionary, window_size=self.window_size,
            coherence=self.coherence, topn=self.topn
        )

        return cm.get_coherence()


class PerplexityMetric(Metric):
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(PerplexityMetric, self).set_parameters(**kwargs)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = self.model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)


class DiffMetric(Metric):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(DiffMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return diff_diagonal


class ConvergenceMetric(Metric):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(ConvergenceMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return np.sum(diff_diagonal)


class Callback:
    def __init__(self, metrics):
        self.metrics = metrics

    def set_model(self, model):
        self.model = model
        self.previous = None
        if any(isinstance(metric, (DiffMetric, ConvergenceMetric)) for metric in self.metrics):
            self.previous = copy.deepcopy(model)
            self.diff_mat = Queue()
        if any(metric.logger == "visdom" for metric in self.metrics):
            if not VISDOM_INSTALLED:
                raise ImportError("Please install Visdom for visualization")
            self.viz = Visdom()
            self.windows = []
        if any(metric.logger == "shell" for metric in self.metrics):
            self.log_type = logging.getLogger('gensim.models.ldamodel')

    def on_epoch_end(self, epoch, topics=None):
        current_metrics = {}
        for i, metric in enumerate(self.metrics):
            label = str(metric)
            value = metric.get_value(topics=topics, model=self.model, other_model=self.previous)
            current_metrics[label] = value
            if metric.logger == "visdom":
                if epoch == 0:
                    if value.ndim > 0:
                        diff_mat = np.array([value])
                        viz_metric = self.viz.heatmap(
                            X=diff_mat.T, env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        self.diff_mat.put(diff_mat)
                        self.windows.append(copy.deepcopy(viz_metric))
                    else:
                        viz_metric = self.viz.line(
                            Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env,
                            opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        self.windows.append(copy.deepcopy(viz_metric))
                else:
                    if value.ndim > 0:
                        diff_mat = np.concatenate((self.diff_mat.get(), np.array([value])))
                        self.viz.heatmap(
                            X=diff_mat.T, env=metric.viz_env, win=self.windows[i],
                            opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        self.diff_mat.put(diff_mat)
                    else:
                        self.viz.line(
                            Y=np.array([value]),
                            X=np.array([epoch]),
                            env=metric.viz_env,
                            win=self.windows[i],
                            update='append'
                        )
            if metric.logger == "shell":
                statement = "".join(("Epoch ", str(epoch), ": ", label, " estimate: ", str(value)))
                self.log_type.info(statement)
        if any(isinstance(metric, (DiffMetric, ConvergenceMetric)) for metric in self.metrics):
            self.previous = copy.deepcopy(self.model)
        return current_metrics


class CallbackAny2Vec:
    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_train_begin(self, model):
        pass

    def on_train_end(self, model):
        pass

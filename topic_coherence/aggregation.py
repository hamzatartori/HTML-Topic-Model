import logging
import numpy as np

logger = logging.getLogger(__name__)


def arithmetic_mean(confirmed_measures):
    return np.mean(confirmed_measures)

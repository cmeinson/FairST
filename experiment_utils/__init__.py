from .file_reader import ResultsReader
from .grapher import ResultsGrapher
from .other_col_reader import OtherColReader

def metric_has_substring(metric, substrings):
    for sub in substrings:
        if sub in metric:
            return True
    return False
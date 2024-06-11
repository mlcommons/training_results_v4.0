from mlperf_logging import mllog
from mlperf_logging.mllog.constants import CACHE_CLEAR
mllogger = mllog.get_mllogger()
mllogger.event(key=CACHE_CLEAR, value=True)


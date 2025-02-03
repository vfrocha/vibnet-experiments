import logging 
import sys
import coloredlogs

# CONFIGURING LOG
logger = logging.getLogger()

# create console handler and set level to info
ch = logging.StreamHandler(sys.stdout)

# create formatter
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# add color to logs
coloredlogs.install(logger=logger)
logger.propagate = False

logger.setLevel(logging.INFO)
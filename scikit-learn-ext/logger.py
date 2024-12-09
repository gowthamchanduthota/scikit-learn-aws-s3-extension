import logging
import os
import time


def logger17():
    curr_script = os.path.basename(__file__)
    logging.basicConfig(
        filename="{}.log".format(curr_script),
        level=logging.INFO,
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    return logger


logger = logger17()
logger.info("Printing list: ")
logger.error("Failed test case: {}".format("error - e"))
# s3_helper.upload_file(file_name = "main.py.log", s3_key="test-s3-poc/{}/main.py.log".format(time.strftime("%Y%m%d-%H%M%S")))
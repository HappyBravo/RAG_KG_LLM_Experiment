import logging
from datetime import datetime

class Logger:
    def __init__(self, log_file="logfile.txt"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Create file handler which logs messages
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

if __name__ == "__main__":
    log_filename = "./Logs/" + f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file=log_filename)
    message = """This is multiline message ... 
    Lets see how this is stored.
    <<okay ??? >>> okat !!! 
    
    noice 👍
    """
    logger.log("Testing ... ")

    logger.log(message)

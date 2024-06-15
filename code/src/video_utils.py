import logging
def set_logging_level(filename):
    """
    Set logging level based on the filename.

    Args:
        filename (str): The name of the file being processed.
    """
    # Extract logging level from filename
    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = filename.split("_")[-1].split(".")[0].upper()

    # Check if the extracted level is valid
    if level in logging_levels:
        logging.basicConfig(level=logging_levels[level])
    else:
        logging.basicConfig(level=logging.INFO)  # Default to INFO level if level is not recognized

    return logging.getLogger(filename)

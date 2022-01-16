import logging
def init_logger(log_dir=None, handler='both'):

    """Initialize logger.
    logging.getLogger(name): Multiple calls to getLogger() with the same name will always return a reference to the same Logger object. (in Python docs)
    So if you call 'logging.getLogger(same_name)' in other codes, it will return a reference to the same Logger object.
    
    To log message, use these methods; logger.info(msg) / logger.warning(msg) / logger.error(msg)
    Args:
        log_dir: if handler is set 'file' or 'both' the logs will be saved at log_dir. Also it is used to identify unique logger (str, optional).
        handler: print the logs at designated places. file: txt / stream: console / both: file and stream. defaults to 'both' (str, optional).
    Returns:
        logger instance
    """

    assert handler in ['file', 'stream', 'both']
    if not log_dir:
        log_dir = os.path.abspath('')
    check_dir(log_dir)

    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO) # message below the setLevel will be ignored.
    # Formatter; the format of the log(print)
    formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')

    if (handler == 'file') or (handler == 'both'):
        init_time = datetime.datetime.now().strftime('%Y.%m.%d %H-%M-%S')
        fname = f'log_{init_time}.log'
        file = os.path.join(log_dir, fname)

        file_handler = logging.FileHandler(filename = file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if (handler == 'stream') or (handler == 'both'):
        # Stream(Console) Handler; Handler object dispatchs specific message for proper levels to specific point like a console or a file.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
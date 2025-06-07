import logging
import os
import colorlog

# Custom logging levels
FUNCTION_CALL_LEVEL = 25
logging.addLevelName(FUNCTION_CALL_LEVEL, 'FUNCTION_CALL')

EVALUATION_LEVEL = 26
logging.addLevelName(EVALUATION_LEVEL, 'EVALUATION')

ANNOUNCEMENT_LEVEL = 27
logging.addLevelName(ANNOUNCEMENT_LEVEL, 'ANNOUNCEMENT')

def function_call(self, message):
    self.log(FUNCTION_CALL_LEVEL, message)

def evaluation(self, message):
    self.log(EVALUATION_LEVEL, message)

def announcement(self, message):
    self.log(ANNOUNCEMENT_LEVEL, message)

logging.Logger.function_call = function_call
logging.Logger.evaluation = evaluation
logging.Logger.announcement = announcement

def setup_logging(run_folder: str, log_level: str = "INFO") -> str:
    """
    Set up logging configuration for the benchmark.
    
    Args:
        run_folder: Base folder for saving logs
        log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Path to the log file
    """
    os.makedirs(f"{run_folder}/logs", exist_ok=True)
    log_file = f"{run_folder}/logs/osintbench.log"
    
    # Set root logger to DEBUG (most permissive)
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green', 
                'FUNCTION_CALL': 'blue',
                'EVALUATION': 'purple',
                'ANNOUNCEMENT': 'bold_green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            style='%'
        )
    except ImportError:
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - always DEBUG level (captures everything, no colors)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - user-specified level (with colors if available)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the custom function_call method.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
import threading

_context = threading.local()

def set_case(case):
    _context.case = case

def get_case():
    return getattr(_context, 'case', None)

def set_dataset_path(dataset_path):
    _context.dataset_path = dataset_path

def get_dataset_path():
    return getattr(_context, 'dataset_path', None)

def set_benchmark(benchmark):
    _context.benchmark = benchmark

def get_benchmark():
    return getattr(_context, 'benchmark', None)
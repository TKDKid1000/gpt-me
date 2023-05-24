from time import time


def timer(func):
    def time_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Function `{func.__name__}` completed in {time()-start} seconds.")
        return result

    return time_func

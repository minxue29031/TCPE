import functools


def retry(retries=3, exceptions=(Exception,)):
    """
    A decorator that retries a function if it raises an exception.

    Parameters:
    retries (int): Number of times to retry the function.
    exceptions (tuple): Tuple of exceptions to catch and retry on.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts < retries:
                        print(f"Attempt {attempts} failed: {e}. Retrying...")
                    else:
                        print(f"Attempt {attempts} failed: {e}. No more retries.")
                        raise

        return wrapper

    return decorator

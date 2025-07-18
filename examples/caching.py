import functools
import hashlib
import os
import pickle


def dict_to_hash(my_dict: dict) -> str:
    """
    Creates a hash from the given dictionary to be used as a unique ID.

    :param my_dict: Dictionary, should not be nested.
    :type my_dict: dict

    :return: Hash value of the given dictionary in string format.
    :rtype: str
    """
    sorted_items = sorted(my_dict.items())
    str_repr = repr(sorted_items).encode("utf-8")
    hash_object = hashlib.md5(str_repr)
    hash_str = hash_object.hexdigest()
    return hash_str


def generate_function_key(fn):
    """
    Generates a key for this callable by hashing the bytecode. This appears
    to be deterministic on CPython for trivial implementations, but likely
    is implementation-specific.

    :param fn: Function to generate the key for.
    :type fn: callable

    :return: Hash value of the function in string format.
    :rtype: str
    """
    return hashlib.md5(fn.__code__.co_code).hexdigest()


def cache_results(memo_filename=None):
    """
    Decorator to cache the results of a function call to a pickle file.

    :param memo_filename: Name of the pickle file to store the results in.
    :type memo_filename: str, optional

    :return: Decorated function.
    :rtype: callable
    """

    def decorator(fn, filename=None):
        if filename is None:
            filename = generate_function_key(fn) + ".pkl"
        base_path = os.path.join(os.path.dirname(__file__), "caches")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, "rb") as fr:
                cache = pickle.load(fr)
                print(f"Loaded cache from {file_path}.")
        except IOError:
            cache = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.update(zip(fn.__code__.co_varnames, args))
            key = dict_to_hash(kwargs)
            if key not in cache:
                cache[key] = fn(**kwargs)
                with open(file_path, "wb") as fw:
                    pickle.dump(cache, fw)
            return cache[key]

        return wrapper

    return functools.partial(decorator, filename=memo_filename)

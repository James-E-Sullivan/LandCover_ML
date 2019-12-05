import pickle


def write_object(obj, filename):
    """
    Write object with pickle.
    :param obj: Object to write to disk.
    :param filename: Output filepath
    """
    with open(filename, 'wb', buffering=2000000) as f:
        pickle.dump(obj, f, protocol=4)   # protocol=4 should allow >4gb


def read_object(filename):
    """
    Read object from disk with pickle.
    :param filename: Input filepath
    :return obj: object loaded from pickle
    """
    with open(filename, 'rb', buffering=2000000) as f:
        obj = pickle.load(f)
        return obj


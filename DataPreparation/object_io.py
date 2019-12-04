import pickle
import json


def write_object(obj, filename):

    with open(filename, 'wb', buffering=2000000) as f:
        pickle.dump(obj, f, protocol=4)   # protocol=4 should allow >4gb


def read_object(filename):

    with open(filename, 'rb', buffering=2000000) as f:
        obj = pickle.load(f)
        return obj


def write_json_object(obj, filename):

    with open(filename, 'w') as f:
        json.dump(obj, f)


def read_json_object(filename):

    with open(filename, 'r') as f:
        obj = json.load(f)
        return obj


if __name__ == '__main__':

    example_obj = [1, 2, 3]
    file_name = 'example_saved_obj.obj'
    json_name = 'example_json_obj.json'

    write_object(example_obj, file_name)
    pickle_object = read_object(file_name)

    write_json_object(example_obj, json_name)
    json_object = read_json_object(json_name)

    print("Pickle Object:", pickle_object)
    print("JSON Object:", json_object)







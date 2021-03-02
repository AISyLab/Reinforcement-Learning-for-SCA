import sys

import h5py
import numpy as np


def load_ches_hd5(database_file,
                  profiling_traces_file, profiling_metadata_file,
                  attack_traces_file, attack_metadata_file):
    try:
        in_file = h5py.File(database_file, "r")
    except ValueError:
        print(f"Error: can't open HDF5 file '{database_file}' for reading (it might be malformed) ...")
        sys.exit(-1)
    # Load profiling traces
    x_profiling = np.array(in_file[profiling_traces_file], dtype=np.float64)
    # Load profiling labels
    metadata_profiling = np.array(in_file[profiling_metadata_file])
    # Load attacking traces
    x_attack = np.array(in_file[attack_traces_file], dtype=np.float64)
    # Load attacking labels
    metadata_attack = np.array(in_file[attack_metadata_file])
    return (x_profiling, metadata_profiling), (x_attack, metadata_attack)


def load_hd5(database_file,
             profiling_traces_file, profiling_labels_file,
             attack_traces_file, attack_labels_file,
             attack_metadata_file=None):
    try:
        in_file = h5py.File(database_file, "r")
    except ValueError:
        print(f"Error: can't open HDF5 file '{database_file}' for reading (it might be malformed) ...")
        sys.exit(-1)
    # Load profiling traces
    x_profiling = np.array(in_file[profiling_traces_file], dtype=np.float64)
    # Load profiling labels
    y_profiling = np.array(in_file[profiling_labels_file])
    # Load attacking traces
    x_attack = np.array(in_file[attack_traces_file], dtype=np.float64)
    # Load attacking labels
    y_attack = np.array(in_file[attack_labels_file])
    if attack_metadata_file is None:
        return (x_profiling, y_profiling), (x_attack, y_attack)
    else:
        return (x_profiling, y_profiling), (x_attack, y_attack), in_file[attack_metadata_file]['plaintext']


def load_hd5_hw_model(database_file,
                      profiling_traces_file, profiling_labels_file,
                      attack_traces_file, attack_labels_file,
                      attack_metadata_file=None):
    result = load_hd5(
        database_file, profiling_traces_file, profiling_labels_file,
        attack_traces_file, attack_labels_file, attack_metadata_file
    )
    y_profiling = np.array([bin(x).count("1") for x in result[0][1]])
    y_attack = np.array([bin(x).count("1") for x in result[1][1]])
    if attack_metadata_file is None:
        return (result[0][0], y_profiling), (result[1][0], y_attack)
    else:
        return (result[0][0], y_profiling), (result[1][0], y_attack), result[2]

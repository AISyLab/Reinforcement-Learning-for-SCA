import multiprocessing as mp
import random
import os
from os import path
from typing import Callable

import matplotlib.pyplot as plot
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

CPU_COUNT = len(os.sched_getaffinity(0)) if 'sched_getaffinity' in dir(os) else mp.cpu_count()


def shuffle_arrays_together(array_a: np.ndarray, array_b: np.ndarray):
    """
    Shuffle two (numpy) arrays while keeping values at the same indices consistent between the two arrays.
    """
    _list = list(zip(array_a, array_b))
    random.shuffle(_list)
    _a, _b = list(zip(*_list))
    return np.array(_a), np.array(_b)


# Compute the position of the key hypothesis key amongst the hypotheses
def rk_key(rank_array, key):
    key_val = rank_array[key]
    return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]


def rank_compute_precomputed_byte_n(prediction, byte_key_hypotheses, key, key_byte):
    number_of_traces, _ = prediction.shape

    key_log_prob = np.zeros(256)
    rank_evol = np.zeros(number_of_traces)
    prediction = np.log(prediction + 1e-40)

    for i in range(number_of_traces):
        for k in range(256):
            key_log_prob[k] += prediction[i, int(byte_key_hypotheses[i, k])]  # Use precomputed hypothesis values

        rank_evol[i] = rk_key(key_log_prob, key[key_byte])

    return rank_evol


# Performs attack
def perform_attacks_precomputed_byte_n_parallel(traces_per_attack: int, predictions, attack_amount,
                                                precomputed_byte0_values, key, key_byte, shuffle=True,
                                                save_graph=False, filename='fig', folder='data') -> np.ndarray:
    all_rk_evol = np.array(Parallel(n_jobs=CPU_COUNT)(delayed(_perform_attack_precomputed_byte_n)(
        traces_per_attack, predictions, precomputed_byte0_values, key, key_byte, shuffle
    ) for _ in tqdm(range(attack_amount))))
    rk_avg = np.mean(all_rk_evol, axis=0)
    _, nb_hyp = predictions.shape

    if save_graph:
        plot_ge(rk_avg, traces_per_attack, attack_amount, filename=filename, folder=folder)

    return rk_avg


def _perform_attack_precomputed_byte_n(traces_per_attack: int, predictions, precomputed_byte0_values,
                                       key, key_byte, shuffle=True):
    if shuffle:
        sp, sbyte0_values = shuffle_arrays_together(predictions, precomputed_byte0_values)
        att_pred = sp[:traces_per_attack]
        att_byte0_values = sbyte0_values[:traces_per_attack]
    else:
        att_pred = predictions[:traces_per_attack]
        att_byte0_values = precomputed_byte0_values[:traces_per_attack]

    return rank_compute_precomputed_byte_n(att_pred, att_byte0_values, key, key_byte)


def plot_ge(rk_avg, traces_per_attack, attack_amount, filename='fig', folder='data'):
    plot.rcParams['figure.figsize'] = (20, 10)
    plot.ylim(-5, 200)
    plot.xlim(0, traces_per_attack + 1)
    plot.grid(True)
    plot.plot(range(1, traces_per_attack + 1), rk_avg, '-')
    plot.xlabel('Number of traces')
    plot.ylabel('Mean rank of correct key guess')

    plot.title(
        f'{filename} Guessing Entropy\nUp to {traces_per_attack:d} traces averaged over {attack_amount:d} attacks',
        loc='center'
    )

    plot.savefig(
        path.normpath(path.join(folder, f'{filename}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')),
        format='svg', dpi=1200, bbox_inches='tight'
    )
    plot.close()


##############################################################################################
# Code not directly used in the RL experiment, but might still be useful in other situations #
##############################################################################################
#####################
#  SBOX Definition  #
#####################

AES_SBox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


###############
#  FUNCTIONS  #
###############
def generate_precomputed_values_file(
        plaintext: np.ndarray, save_folder: str, byte: int = 0,
        mapping: Callable[[np.ndarray, np.ndarray, np.ndarray, int], int] =
        lambda i, j, plaintext, byte: AES_SBox[j ^ plaintext[i, byte]]) -> np.ndarray:
    """
    Generate precomputed byte values file from the plaintext combined with all possible key values
    :param plaintext: array-like
    The plaintext values for each trace (expected shape: (#traces, 16)
    :param save_folder: str
    The folder to save the precomputed values in
    :param byte: int
    The byte of the key to generate the value for (determines the plaintext segment to use)
    :param mapping: lambda i, j, plaintext, byte
    In case a custom mapping is required (e.g., in the presence of masking)
    :return: The precomputed byte values ndarray
    """
    # noinspection PyTypeChecker
    precomputed: np.ndarray = np.fromfunction(
        lambda i, j: mapping(i, j, plaintext.astype(int), int(byte)),
        (plaintext.shape[0], AES_SBox.shape[0]),
        dtype=int
    )
    if save_folder is not None:
        np.save(path.normpath(f'{save_folder}/attack_precomputed_byte{byte}_values.npy'), precomputed)

    return precomputed


def vectorized_unmask_sbox(trace_i: np.ndarray, key_byte: np.ndarray, plaintext: np.ndarray,
                           byte_i: int, mask: np.ndarray, offset: np.ndarray):
    """
    Function to compute leaking intermediary values in the case of a masked implementation with offset
    in a vectorized manner (for performance).
    This implementation can for example be used for the generate_precomputed_values_file() function.
    """
    return AES_SBox[key_byte ^ plaintext[trace_i, byte_i]] ^ mask[((offset[trace_i, byte_i] + 1) % 16).astype(int)]


def rank_compute_precomputed(prediction, key_hypotheses, key, byte):
    (nb_trs, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs, 255)
    prediction = np.log(prediction + 1e-40)

    for i in range(nb_trs):
        for k in range(nb_hyp):
            key_log_prob[k] += prediction[i, int(key_hypotheses[i, byte, k])]  # Use precomputed hypothesis values

        rank_evol[i] = rk_key(key_log_prob, key[byte])

    return rank_evol


# Compute the evolution of rank
def rank_compute_unmask(prediction, att_plt, key, mask, offset, byte):
    """
    :param prediction: predictions of the NN
    :param att_plt: plaintext of the attack traces
    :param key: Key used during encryption
    :param mask: the known mask used during the encryption
    :param offset: offset used for computing the Sbox output
    :param byte: byte to attack
    :return: The rank of the correct key guess for of the traces
    """

    (nb_trs, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs, 255)
    prediction = np.log(prediction + 1e-40)

    for i in range(nb_trs):
        for k in range(nb_hyp):
            key_log_prob[k] += prediction[i, (
                    AES_SBox[int(att_plt[i, byte]) ^ int(k)] ^ int(mask[int(offset[i, byte] + 1) % 16])
            )]  # Computes the hypothesis values

        rank_evol[i] = rk_key(key_log_prob, key[byte])

    return rank_evol


# Performs attack
def perform_attacks_precomputed_byte_n(traces_per_attack: int, predictions, attack_amount,
                                       precomputed_byte0_values, key, key_byte, shuffle=True,
                                       save_graph=False, filename='fig', folder='data') -> np.ndarray:
    all_rk_evol = np.zeros((attack_amount, traces_per_attack))
    for i in range(attack_amount):
        all_rk_evol[i] = _perform_attack_precomputed_byte_n(
            traces_per_attack, predictions, precomputed_byte0_values, key, key_byte, shuffle
        )

    rk_avg = np.mean(all_rk_evol, axis=0)
    _, nb_hyp = predictions.shape

    if save_graph:
        plot_ge(rk_avg, traces_per_attack, attack_amount, filename=filename, folder=folder)

    return rk_avg


def perform_attacks(traces_per_attack: int, predictions, attack_amount: int, plaintexts, key, mask, offsets, byte=0,
                    shuffle=True, save_graph=False, filename='fig') -> np.ndarray:
    """
    Performs a given number of attacks to be determined

    :param traces_per_attack: number of traces to use to perform each attack
    :param predictions: array containing the values of the model predictions
    :param attack_amount: number of attack to perform
    :param plaintexts: the plaintexts used to obtain the consumption traces
    :param key: the key used to obtain the consumption traces
    :param mask: the known mask used during the encryption
    :param offsets: SBox output mask offsets used for each of the traces
    :param byte: byte to attack
    :param shuffle: Whether the traces have to be shuffled (Default = True)
    :param save_graph: Whether or not to save a GE graph
    :param filename: The filename of the GE graph (Default = fig)
    :return: The average GE over all attacks for each amount of traces per attack
    """

    # (nb_total, nb_hyp) = predictions.shape

    all_rk_evol = np.zeros((attack_amount, traces_per_attack))
    for i in range(attack_amount):
        if shuffle:
            _list = list(zip(predictions, plaintexts, offsets))
            random.shuffle(_list)
            sp, splt, soffset = list(zip(*_list))
            sp = np.array(sp)
            splt = np.array(splt)
            soffset = np.array(soffset)
            att_pred = sp[:traces_per_attack]
            att_plt = splt[:traces_per_attack]
            att_offset = soffset[:traces_per_attack]

        else:
            att_pred = predictions[:traces_per_attack]
            att_plt = plaintexts[:traces_per_attack]
            att_offset = offsets[:traces_per_attack]

        rank_evolution = rank_compute_unmask(att_pred, att_plt, key, mask, att_offset, byte=byte)
        all_rk_evol[i] = rank_evolution

    rk_avg = np.mean(all_rk_evol, axis=0)

    if save_graph:
        plot_ge(rk_avg, traces_per_attack, attack_amount, filename=filename)

    return rk_avg

import tensorflow as tf
import subprocess


def debug():
    phy_gpu = tf.config.experimental.list_physical_devices('GPU')
    log_gpu = tf.config.experimental.list_logical_devices('GPU')
    lspci = subprocess.run(['lspci', '-vnnn'], stdout=subprocess.PIPE)
    lspci = subprocess.run(['grep', '-i', "nvidia"], input=lspci.stdout, stdout=subprocess.PIPE)

    print("\n\n\n")
    print("Physical GPU devices:")
    print(phy_gpu)

    print("\n\nLogical GPU devices:")
    print(log_gpu)

    print("\n\nlspci -vnnn | grep -i 'nvidia':")
    print(lspci.stdout.decode('utf-8'))


if __name__ == "__main__":
    debug()

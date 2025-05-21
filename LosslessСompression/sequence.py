import random
import string
import collections
import math
import os
import matplotlib.pyplot as plt

surname = "вілейський"
group_number = "529"
student_index = 4
N_sequence = 100

folder = os.path.join("..", "LosslessСompression")
os.makedirs(folder, exist_ok=True)


def save_sequence_info(seq_num, sequence, path):
    unique_chars = set(sequence)
    alphabet_size = len(unique_chars)
    size_bytes = len(sequence)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Original sequence {seq_num}: {sequence}\n")
        f.write(f"Sequence {seq_num} alphabet size: {alphabet_size}\n")
        f.write(f"Original sequence {seq_num} size (bytes): {size_bytes}\n\n")

def sequence1():
    l1 = ['1'] * student_index
    l0 = ['0'] * (N_sequence - student_index)
    seq = l1 + l0
    random.shuffle(seq)
    return ''.join(seq)

def sequence2():
    l1 = list(surname)
    l0 = ['0'] * (N_sequence - len(l1))
    return ''.join(l1 + l0)

def sequence3():
    n_letters = len(surname)
    letters = [random.choice(list(surname)) for _ in range(n_letters)]
    zeros = ['0'] * (N_sequence - n_letters)
    seq = letters + zeros
    random.shuffle(seq)
    return ''.join(seq)

def sequence4():
    letters = list(surname) + list(group_number)
    n = len(letters)
    repeats = N_sequence // n
    rem = N_sequence % n
    seq = letters * repeats + letters[:rem]
    return ''.join(seq)

def sequence5():
    alphabet = list(surname[:2]) + list(group_number)
    seq = [random.choice(alphabet) for _ in range(N_sequence)]
    random.shuffle(seq)
    return ''.join(seq)

def sequence6():
    letters = list(surname[:2])
    digits = list(group_number)
    n_letters = int(0.7 * N_sequence)
    n_digits = N_sequence - n_letters
    seq = [random.choice(letters) for _ in range(n_letters)] + [random.choice(digits) for _ in range(n_digits)]
    random.shuffle(seq)
    return ''.join(seq)

def sequence7():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(N_sequence))

def sequence8():
    return '1' * N_sequence

sequences = [sequence1(), sequence2(), sequence3(), sequence4(), sequence5(), sequence6(), sequence7(), sequence8()]

with open(f"{folder}/sequence.txt", "w", encoding="utf-8") as f:
    for seq in sequences:
        f.write(seq + "\n")

results_table = []
with open(f"{folder}/results_sequence.txt", "w", encoding="utf-8") as f:
    for i, seq in enumerate(sequences, start=1):
        save_sequence_info(i, seq, f"{folder}/results_sequence.txt")
        counts = collections.Counter(seq)
        probability = {s: c / N_sequence for s, c in counts.items()}
        mean_p = sum(probability.values()) / len(probability)
        uniformity = all(abs(p - mean_p) < 0.05 * mean_p for p in probability.values())
        entropy = -sum(p * math.log2(p) for p in probability.values())
        alphabet_size = len(probability)
        redundancy = 1 - entropy / math.log2(alphabet_size) if alphabet_size > 1 else 1
        prob_str = ', '.join(f"{s}={p:.4f}" for s, p in probability.items())
        f.write(f"Analysis for sequence {i}:\n")
        f.write(f"Probabilities: {prob_str}\n")
        f.write(f"Entropy: {entropy:.4f}\n")
        f.write(f"Redundancy: {redundancy:.4f}\n")
        f.write(f"Type: {'рівна' if uniformity else 'нерівна'}\n\n")
        results_table.append([alphabet_size, round(entropy, 2), round(redundancy, 2), 'рівна' if uniformity else 'нерівна'])

fig, ax = plt.subplots(figsize=(14/1.54, len(sequences)/1.54))
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Тип']
rows = [f"Послідовність {i}" for i in range(1, len(sequences)+1)]
ax.axis('off')
table = ax.table(cellText=results_table, colLabels=headers, rowLabels=rows, loc='center', cellLoc='center')
table.set_fontsize(14)
table.scale(0.8, 2)
fig.savefig(f"{folder}/Характеристики сформованих послідовностей.png", bbox_inches='tight')

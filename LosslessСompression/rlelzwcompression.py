import collections
import math
import matplotlib.pyplot as plt
from collections import Counter
from math import log2

def read_sequences(path):
    with open(path, "r", encoding="utf-8") as f:
        seqs = [line.strip() for line in f if line.strip()]
    return seqs

def compute_stats(sequence):
    N = len(sequence)
    counts = collections.Counter(sequence)
    probs = {sym: cnt / N for sym, cnt in counts.items()}
    entropy = -sum(p * math.log2(p) for p in probs.values())
    bits_original = N * 16
    return counts, probs, entropy, bits_original

def encode_rle(sequence):
    if not sequence:
        return "", []
    result = []
    count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            count += 1
        else:
            result.append((sequence[i-1], count))
            count = 1
    result.append((sequence[-1], count))
    encoded_str = "".join(f"{cnt}{sym}" for sym, cnt in result)
    return encoded_str, result

def compute_cr(original_len, encoded_len):
    if encoded_len == 0:
        return "-"
    cr = round(original_len / encoded_len, 2)
    return cr if cr >= 1 else "-"

def decode_rle(sequence, index):
    result = []
    for count, char in sequence:
        result.append(char * count)
    decoded = "".join(result)
    with open("results_rle_lzw.txt", "a", encoding='utf-8') as file:
        file.write("\n\n\n\n")
        file.write(f"Послідовність {index}\n")
        file.write("\n")
        file.write(f"Декодована RLE послідовність: {decoded}\n")
        file.write(f"Розмір декодованої RLE послідовності: {len(decoded) * 8} bits\n")
    return decoded


def encode_lzw(sequence):
    dictionary = {chr(i): i for i in range(65536)}
    current = ""
    result = []
    size = 0

    with open("results_rle_lzw.txt", "a", encoding='utf-8') as file:
        for char in sequence:
            new_str = current + char
            if new_str in dictionary:
                current = new_str
            else:
                code = dictionary[current]
                result.append(code)
                element_bits = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
                file.write(f"Code: {code}, Element: {current}, Bits: {element_bits}\n")
                size += element_bits
                dictionary[new_str] = len(dictionary)
                current = char

        if current:
            code = dictionary[current]
            result.append(code)
            last_bits = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
            file.write(f"Code: {code}, Element: {current}, Bits: {last_bits}\n")
            size += last_bits

        file.write(f"Закодована LZW послідовність: {''.join(map(str, result))}\n")
        file.write(f"Розмір закодованої LZW послідовності: {size} bits\n")

    return result, size

def decode_lzw(sequence):
    dictionary = {i: chr(i) for i in range(65536)}
    result = ""
    previous = None

    for code in sequence:
        if code in dictionary:
            current = dictionary[code]
        else:
            current = previous + previous[0]

        result += current

        if previous is not None:
            dictionary[len(dictionary)] = previous + current[0]

        previous = current

    with open("results_rle_lzw.txt", "a", encoding='utf-8') as file:
        file.write(f"Декодована LZW послідовність: {result}\n")
        file.write(f"Розмір декодованої LZW послідовності: {len(result) * 16} bits\n")

    return result

def calculate_entropy(sequence):
    total = len(sequence)
    frequencies = Counter(sequence)
    entropy = -sum((freq/total) * log2(freq/total) for freq in frequencies.values())
    return entropy

def rle_encode(sequence):
    result = []
    prev = sequence[0]
    count = 1
    for char in sequence[1:]:
        if char == prev:
            count += 1
        else:
            result.append((count, prev))
            prev = char
            count = 1
    result.append((count, prev))
    return result

def run_all(sequences):
    results = []
    N = len(sequences)

    for i, seq in enumerate(sequences):
        entropy = calculate_entropy(seq)
        rle_encoded = rle_encode(seq)
        rle_encoded_size = sum([8 + 8 for _ in rle_encoded])
        rle_decoded = decode_rle(rle_encoded, i + 1)
        compression_ratio_RLE = round((len(seq) * 8) / rle_encoded_size, 2)
        lzw_encoded, lzw_encoded_size = encode_lzw(seq)
        lzw_decoded = decode_lzw(lzw_encoded)
        compression_ratio_LZW = round((len(seq) * 16) / lzw_encoded_size, 2)
        results.append([round(entropy, 2), compression_ratio_RLE, compression_ratio_LZW])

    fig, ax = plt.subplots(figsize=(14/1.54, N/1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    rows = [f"Послідовність {i+1}" for i in range(N)]
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=rows, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig("Результати стиснення методами RLE та LZW.png")

sequences = read_sequences("sequence.txt")
with open("results_rle_lzw.txt", "w", encoding="utf-8") as out:
    for idx, seq in enumerate(sequences, 1):
        counts, probs, entropy, bits_orig = compute_stats(seq)
        encoded, runs = encode_rle(seq)
        bits_enc = len(encoded) * 16
        cr = compute_cr(len(seq), len(encoded))

        out.write(f"Оригінальна послідовність: {seq}\n")
        out.write(f"Розмір оригінальної послідовості: {bits_orig} bits\n")
        out.write(f"Ентропія: {entropy:.4f}\n")
        out.write("-----  Кодування_RLE:  -----\n")
        out.write(f"Закодована RLE послідовність: {encoded}\n")
        out.write(f"Розмір закодованої RLE послідовності: {bits_enc} bits\n")
        out.write(f"Коефіцієнт стиснення RLE: {cr}\n")
        out.write(f"Кількість символів: {dict(counts)}\n")
        out.write(f"Ймовірності: {probs}\n")
        out.write(f"Runs: {runs}\n")
        out.write("\n")


run_all(sequences)

import collections
import math
import re
import matplotlib.pyplot as plt

def float_bin(point, size_cod):
    binary_code = ''
    for _ in range(size_cod):
        point *= 2
        if point == 1:
            binary_code += '1'
            break
        if point > 1:
            binary_code += '1'
            point -= int(point)
        else:
            binary_code += '0'
    return binary_code

def encode_ac(uniq_chars, probabilitys, alphabet_size, sequence):
    if alphabet_size == 1:
        return [0.5, 1, list(uniq_chars), [1.0]], '0'
    alphabet = list(uniq_chars)
    probability = [probabilitys[s] for s in alphabet]
    unity = []
    pr = 0.0
    for i, p in enumerate(probability):
        l = pr
        pr += p
        unity.append([alphabet[i], l, pr])
    for i in range(len(sequence) - 1):
        for j in range(len(unity)):
            if sequence[i] == unity[j][0]:
                low, high = unity[j][1], unity[j][2]
                diff = high - low
                for k in range(len(unity)):
                    unity[k][1] = low
                    unity[k][2] = probability[k] * diff + low
                    low = unity[k][2]
                break
    # last symbol
    for sym, l, u in unity:
        if sym == sequence[-1]:
            low, high = l, u
            break
    diff = high - low
    if diff == 0:
        bin_code = '0'
    else:
        size_cod = math.ceil(math.log(1/diff, 2) + 1)
        bin_code = float_bin((low + high)/2, size_cod)
    return [(low+high)/2, alphabet_size, alphabet, probability], bin_code

def decode_ac(encoded_data_ac, length_seq):
    point, alphabet_size, alphabet, probability = encoded_data_ac
    unity = []
    pr = 0.0
    for i, p in enumerate(probability):
        l = pr
        pr += p
        unity.append([alphabet[i], l, pr])
    decoded = ''
    for _ in range(length_seq):
        for j in range(len(unity)):
            if unity[j][1] < point < unity[j][2]:
                decoded += unity[j][0]
                pl, ph = unity[j][1], unity[j][2]
                diff = ph - pl
                for k in range(len(unity)):
                    unity[k][1] = pl
                    unity[k][2] = probability[k] * diff + pl
                    pl = unity[k][2]
                break
    return decoded

def encode_ch(uniq_chars, probabilitys, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[s] for s in alphabet]
    final = [[alphabet[i], probability[i]] for i in range(len(alphabet))]
    final.sort(key=lambda x: x[1])
    if len(final) == 1:
        symbol_code = [[final[0][0], '0']]
    else:
        tree = []
        while len(final) > 1:
            left = final.pop(0)
            right = final.pop(0)
            tree.append([left[0], right[0]])
            final.append([left[0]+right[0], left[1]+right[1]])
            final.sort(key=lambda x: x[1])
        tree.reverse()
        alphabet.sort()
        symbol_code = []
        for sym in alphabet:
            code = ''
            for node in tree:
                if sym in node[0]:
                    code += '0'
                    if sym == node[0]: break
                else:
                    code += '1'
                    if sym == node[1]: break
            symbol_code.append([sym, code])
    encode = ''.join(next(c for s,c in symbol_code if s==ch) for ch in sequence)
    return [encode, symbol_code], encode

def decode_ch(encoded_sequence):
    encode_list = list(encoded_sequence[0])
    symbol_code = encoded_sequence[1]
    inv = {c:s for s,c in symbol_code}
    sequence = ''
    buff = ''
    for b in encode_list:
        buff += b
        if buff in inv:
            sequence += inv[buff]
            buff = ''
    return sequence


def main():
    with open('sequence.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()
    original = re.findall(r"[01]+", text) if text.startswith('[') else [line.strip() for line in text.splitlines() if
                                                                        line.strip()]
    N_sequence = len(original[0])
    results = []
    with open('results_AC_CH.txt', 'w', encoding='utf-8') as out:
        for idx, seq in enumerate(original, 1):
            seq10 = seq[:10]
            uniq = set(seq10)
            counts = collections.Counter(seq10)
            prob = {s: counts[s] / N_sequence for s in uniq}
            ent = -sum(p * math.log2(p) for p in prob.values())
            out.write(f"Оригінальна послідовність {idx}: {seq10}\n")
            out.write(f"Ентропія: {ent:.4f}\n\n")
            out.write("——— Арифметичне кодування: ———\n")
            data_ac, enc_ac = encode_ac(uniq, prob, len(uniq), seq10)
            dec_ac = decode_ac(data_ac, len(seq10))
            bps_ac = len(enc_ac) / len(seq10)
            out.write("Дані закодованої АС послідовності: " + str(data_ac) + "\n")
            out.write("Закодована АС послідовність: " + enc_ac + "\n")
            out.write(f"Значення bps при кодуванні АС: {bps_ac:.4f}\n")
            out.write("Декодована АС послідовність: " + dec_ac + "\n\n")
            out.write("——— Кодування Хаффмана: ———\n")
            data_ch, enc_ch = encode_ch(uniq, prob, seq10)
            dec_ch = decode_ch(data_ch)
            bps_ch = len(enc_ch) / len(seq10)
            out.write("Алфавіт      Код символу\n")
            for s, c in data_ch[1]:
                out.write(f"{s}             {c}\n")
            out.write("\n")
            out.write("Дані закодованої НС послідовності: " + str(data_ch) + "\n")
            out.write("Закодована НС послідовність: " + enc_ch + "\n")
            out.write(f"Значення bps при кодуванні НС: {bps_ch:.4f}\n")
            out.write("Декодована НС послідовність: " + dec_ch + "\n\n\n\n\n\n\n")

            results.append((ent, bps_ac, bps_ch))

    fig, ax = plt.subplots(figsize=(14 / 1.54, len(original) / 1.54))
    ax.axis('off')
    table = ax.table(
        cellText=[[round(x[0], 2), x[1], x[2]] for x in results],
        colLabels=['Ентропія', 'bps AC', 'bps CH'],
        rowLabels=[f'Послідовність {i + 1}' for i in range(len(original))],
        loc='center', cellLoc='center'
    )
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig('Результати стиснення методами AC та CH.png')


if __name__=='__main__':
    main()

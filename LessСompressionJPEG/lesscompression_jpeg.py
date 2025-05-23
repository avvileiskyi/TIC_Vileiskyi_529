import os
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image

def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')

def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)

def load_quantization_table(component):
    if component == 'lum':
        q = np.array([
            [2, 2, 2, 2, 3, 4, 5, 6],
            [2, 2, 2, 2, 3, 4, 5, 6],
            [2, 2, 2, 2, 4, 5, 7, 9],
            [2, 2, 2, 4, 5, 7, 9, 12],
            [3, 3, 4, 5, 8, 10, 12, 12],
            [4, 4, 5, 7, 10, 12, 12, 12],
            [5, 5, 7, 9, 12, 12, 12, 12],
            [6, 6, 9, 12, 12, 12, 12, 12]
        ])
    elif component == 'chrom':
        q = np.array([
            [3, 3, 5, 9, 13, 15, 15, 15],
            [3, 4, 6, 11, 14, 12, 12, 12],
            [5, 6, 9, 14, 12, 12, 12, 12],
            [9, 11, 14, 12, 12, 12, 12, 12],
            [13, 14, 12, 12, 12, 12, 12, 12],
            [15, 12, 12, 12, 12, 12, 12, 12],
            [15, 12, 12, 12, 12, 12, 12, 12],
            [15, 12, 12, 12, 12, 12, 12, 12]
        ])
    else:
        raise ValueError(f"component must be 'lum' or 'chrom', got '{component}'")
    return q

def zigzag_points(rows, cols):
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)
    def move(direction, p):
        return {
            UP: lambda pt: (pt[0]-1, pt[1]),
            DOWN: lambda pt: (pt[0]+1, pt[1]),
            LEFT: lambda pt: (pt[0], pt[1]-1),
            RIGHT: lambda pt: (pt[0], pt[1]+1),
            UP_RIGHT: lambda pt: move(UP, move(RIGHT, pt)),
            DOWN_LEFT: lambda pt: move(DOWN, move(LEFT, pt))
        }[direction](p)
    def inbounds(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols
    point = (0, 0)
    move_up = True
    for _ in range(rows * cols):
        yield point
        if move_up:
            nr = move(UP_RIGHT, point)
            if inbounds(nr):
                point = nr
            else:
                move_up = False
                nr = move(RIGHT, point)
                point = nr if inbounds(nr) else move(DOWN, point)
        else:
            nr = move(DOWN_LEFT, point)
            if inbounds(nr):
                point = nr
            else:
                move_up = True
                nr = move(DOWN, point)
                point = nr if inbounds(nr) else move(RIGHT, point)

def block_to_zigzag(block):
    return np.array([block[pt] for pt in zigzag_points(*block.shape)])

def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result

def binstr_flip(s):
    if not set(s).issubset({'0', '1'}):
        raise ValueError("binstr must contain only '0' and '1'")
    return ''.join('0' if c=='1' else '1' for c in s)

def uint_to_binstr(number, size):
    return bin(number)[2:].zfill(size)[-size:]

def int_to_binstr(n):
    if n == 0:
        return ''
    size = bits_required(n)
    if n < 0:
        n -= 1
        n = (1 << size) + n
    return format(n & ((1<<size)-1), f'0{size}b')

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def run_length_encode(arr):
    last = max((i for i,v in enumerate(arr) if v!=0), default=-1)
    symbols, values, run = [], [], 0
    for i,v in enumerate(arr):
        if i>last:
            symbols.append((0,0))
            values.append(int_to_binstr(0))
            break
        if v==0 and run<15:
            run+=1
        else:
            size = bits_required(v)
            symbols.append((run, size))
            values.append(int_to_binstr(v))
            run=0
    return symbols, values

def write_to_file(filepath, dc, ac, blocks_count, tables):
    f = open(filepath, 'w')
    for name in ['dc_y','ac_y','dc_c','ac_c']:
        table = tables[name]
        f.write(uint_to_binstr(len(table), 16))
        for k,v in table.items():
            if name in ('dc_y','dc_c'):
                f.write(uint_to_binstr(k,4))
                f.write(uint_to_binstr(len(v),4))
                f.write(v)
            else:
                f.write(uint_to_binstr(k[0],4))
                f.write(uint_to_binstr(k[1],4))
                f.write(uint_to_binstr(len(v),8))
                f.write(v)
    f.write(uint_to_binstr(blocks_count,32))
    for b in range(blocks_count):
        for c in range(3):
            cat = bits_required(dc[b,c])
            syms, vals = run_length_encode(ac[b,:,c])
            dc_table = tables['dc_y'] if c==0 else tables['dc_c']
            ac_table = tables['ac_y'] if c==0 else tables['ac_c']
            f.write(dc_table[cat])
            f.write(int_to_binstr(dc[b,c]))
            for s,val in zip(syms,vals):
                f.write(ac_table[s])
                f.write(val)
    f.close()

def encode(input_file, output_file, quant_variants=('lum','chrom')):
    image = Image.open(input_file)
    ycbcr = image.convert('YCbCr')
    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[:2]
    if rows%8!=0 or cols%8!=0:
        raise ValueError('width and height must be multiples of 8')
    blocks = (rows//8)*(cols//8)
    dc = np.empty((blocks,3),np.int32)
    ac = np.empty((blocks,63,3),np.int32)
    idx=0
    for i in range(0,rows,8):
        for j in range(0,cols,8):
            for k in range(3):
                block = npmat[i:i+8,j:j+8,k].astype(np.int32)-128
                d = dct_2d(block)
                qm = quantize(d, quant_variants[0] if k == 0 else quant_variants[1])
                zz = block_to_zigzag(qm)
                dc[idx,k] = zz[0]
                ac[idx,:,k] = zz[1:]
            idx+=1
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:,0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:,1:].flat))
    H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i,:,0])[0] for i in range(blocks)))
    H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i,:,j])[0] for i in range(blocks) for j in (1,2)))
    tables = {
        'dc_y': H_DC_Y.value_to_bitstring_table(),
        'ac_y': H_AC_Y.value_to_bitstring_table(),
        'dc_c': H_DC_C.value_to_bitstring_table(),
        'ac_c': H_AC_C.value_to_bitstring_table()
    }
    write_to_file(output_file, dc, ac, blocks, tables)

class JPEGFileReader:
    TABLE_SIZE_BITS=16
    BLOCKS_COUNT_BITS=32
    DC_CODE_LENGTH_BITS=4
    CATEGORY_BITS=4
    AC_CODE_LENGTH_BITS=8
    RUN_LENGTH_BITS=4
    SIZE_BITS=4
    def __init__(self, path):
        self.f = open(path,'r')
    def read_int(self,size):
        if size==0:
            return 0
        b=self.__read_str(size)
        if b[0]=='1':
            return self.__int2(b)
        return -self.__int2(binstr_flip(b))
    def read_dc_table(self):
        t={},
        n=self.__read_uint(self.TABLE_SIZE_BITS)
        table={}
        for _ in range(n):
            cat=self.__read_uint(self.CATEGORY_BITS)
            l=self.__read_uint(self.DC_CODE_LENGTH_BITS)
            c=self.__read_str(l)
            table[c]=cat
        return table
    def read_ac_table(self):
        n=self.__read_uint(self.TABLE_SIZE_BITS)
        table={}
        for _ in range(n):
            rl=self.__read_uint(self.RUN_LENGTH_BITS)
            sz=self.__read_uint(self.SIZE_BITS)
            l=self.__read_uint(self.AC_CODE_LENGTH_BITS)
            c=self.__read_str(l)
            table[c]=(rl,sz)
        return table
    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)
    def read_huffman_code(self,table):
        p=''
        while p not in table:
            p+=self.__read_str(1)
        return table[p]
    def __read_uint(self,size):
        return self.__int2(self.__read_str(size))
    def __read_str(self,n):
        return self.f.read(n)
    def __int2(self,s):
        return int(s,2)

def read_image_file(path):
    reader=JPEGFileReader(path)
    tables={}
    for n in ('dc_y','ac_y','dc_c','ac_c'):
        tables[n]=reader.read_dc_table() if 'dc' in n else reader.read_ac_table()
    blocks=reader.read_blocks_count()
    dc=np.empty((blocks,3),np.int32)
    ac=np.empty((blocks,63,3),np.int32)
    for b in range(blocks):
        for c in range(3):
            dc_t=tables['dc_y'] if c==0 else tables['dc_c']
            ac_t=tables['ac_y'] if c==0 else tables['ac_c']
            cat=reader.read_huffman_code(dc_t)
            dc[b,c]=reader.read_int(cat)
            count=0
            while count<63:
                rl,sz=reader.read_huffman_code(ac_t)
                if (rl,sz)==(0,0):
                    while count<63:
                        ac[b,count,c]=0
                        count+=1
                else:
                    for _ in range(rl):
                        ac[b,count,c]=0
                        count+=1
                    ac[b,count,c]=reader.read_int(sz) if sz>0 else 0
                    count+=1
    return dc,ac,tables,blocks

def zigzag_to_block(zz):
    n=int(math.sqrt(len(zz)))
    if n*n!=len(zz):
        raise ValueError
    b=np.empty((n,n),np.int32)
    for i,pt in enumerate(zigzag_points(n,n)):
        b[pt]=zz[i]
    return b

def dequantize(block,component):
    return block*load_quantization_table(component)

def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T,norm='ortho').T,norm='ortho')

def decoder(path,input_jpeg_name):
    dc,ac,tables,blocks=read_image_file(path)
    side=int(math.sqrt(blocks))*8
    per_line=side//8
    arr=np.empty((side,side,3),np.uint8)
    for b in range(blocks):
        i=(b//per_line)*8
        j=(b%per_line)*8
        for c in range(3):
            zz=[dc[b,c]]+list(ac[b,:,c])
            qm=zigzag_to_block(zz)
            dmat=dequantize(qm,'lum' if c==0 else 'chrom')
            blk=idct_2d(dmat)
            arr[i:i+8,j:j+8,c]=(blk+128)
    img=Image.fromarray(arr,'YCbCr').convert('RGB')
    img.save(input_jpeg_name)
    return img

if __name__ == '__main__':
    os.makedirs('Results', exist_ok=True)
    image_labels = {
        '3_1.bmp': 'сильнотекстурне зображення',
        '3_2.bmp': 'среднетекстурне зображення',
        '3_3.bmp': 'слаботекстурне зображення'
    }
    quant_label = {'lum': 'таблиця квантування 1', 'chrom': 'таблиця квантування 2'}

    report_lines = []

    for infile in ('3_1.bmp', '3_2.bmp', '3_3.bmp'):
        original_size = os.path.getsize(infile)
        img = Image.open(infile)
        width, height = img.size

        for qtype in ('lum', 'chrom'):
            txt_name = f'results_{qtype}_{infile.replace(".bmp", ".txt")}'
            jpg_name = infile.replace('.bmp', f'_{qtype}_decoded.jpg')
            txt_path = os.path.join('Results', txt_name)
            jpg_path = os.path.join('Results', jpg_name)

            encode(infile, txt_path, quant_variants=(qtype, qtype))
            decoder(txt_path, jpg_path)

            jpeg_size = os.path.getsize(jpg_path)
            ratio = original_size / jpeg_size if jpeg_size != 0 else 0

            report_lines.append(
                f"Дані для {image_labels[infile]} ({quant_label[qtype]})\n"
                f"Розмір вихідного файла: {original_size} байт\n"
                f"Розмір файла JPEG: {jpeg_size} байт\n"
                f"Розмір зображення JPEG: {width}x{height}\n"
                f"Коефіцієнт стиснення {ratio:.2f}\n"
            )

    with open('results_jpeg.txt', 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report_lines))

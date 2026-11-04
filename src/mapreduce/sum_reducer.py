#!/usr/bin/env python3
"""
Reducer penjumlah vektor - dipakai sebagai COMBINER dan REDUCER sekaligus
Input (sudah diurutkan per kunci oleh fase shuffle & sort):

    kunci<TAB>v1[<TAB>v2 ...]

Output: satu baris per kunci, tiap kolom nilai dijumlahkan.

    kunci<TAB>sum(v1)[<TAB>sum(v2) ...]

Kenapa satu skrip boleh menjadi combiner DAN reducer: penjumlahan bersifat
asosiatif dan komutatif, jadi menjumlah hasil parsial lalu menjumlahkannya lagi
sama persis dengan menjumlah sekaligus. Combiner berjalan di sisi map dan
memangkas volume yang harus dikirim lewat jaringan saat shuffle.

Rata-rata TIDAK asosiatif (rata-rata dari rata-rata salah bila ukuran grupnya
berbeda), itu sebabnya job yang butuh rata-rata mengeluarkan pasangan
(jumlah, cacah) dan membagi belakangan - bukan rata-rata di dalam reducer.

Kolom bertipe bilangan bulat tetap dicetak sebagai bilangan bulat; kolom yang
mengandung pecahan dicetak dengan presisi penuh (`repr`).
"""

import io
import sys


def _num(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def _emit(out, key, sums):
    out.write(key + "\t" + "\t".join(repr(s) for s in sums) + "\n")


def main():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")

    current, sums = None, []
    for line in stdin:
        key, _, rest = line.rstrip("\n").partition("\t")
        if not rest:
            sys.stderr.write("reporter:counter:novel,malformed_reduce_lines,1\n")
            continue
        values = [_num(v) for v in rest.split("\t")]
        if key != current:
            if current is not None:
                _emit(stdout, current, sums)
            current, sums = key, values
        else:
            sums = [a + b for a, b in zip(sums, values)]
    if current is not None:
        _emit(stdout, current, sums)
    stdout.flush()


if __name__ == "__main__":
    main()

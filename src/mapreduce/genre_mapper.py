#!/usr/bin/env python3
"""
JOB B - FREKUENSI GENRE & KO-OKURENSI atas data SEMI-TERSTRUKTUR (JSONL)
Mapper: satu objek JSON per baris masuk. Keluar dua jenis kunci berprefiks:

    G:<genre>            1     - satu novel memiliki genre ini
    P:<genreA>|<genreB>  1     - dua genre muncul bersama (A < B, tiap pasangan sekali)

Kedua jenis dikerjakan dalam satu job dengan prefiks pada kunci, bukan dua job,
karena keduanya membaca input yang sama: dua job berarti membaca 28 MB dua kali.
Prefiks juga memastikan kedua jenis tidak pernah bertabrakan di reducer.

Genre diurutkan dan didedup per novel supaya {Action, Romance} dan
{Romance, Action} menjadi pasangan yang sama, dan tiap pasangan dihitung SEKALI
per novel (bukan sekali per kemunculan).

Reducer + combiner: `sum_reducer.py`.
"""

import io
import itertools
import json
import sys


def main():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            genres = sorted(set(json.loads(line).get("genres") or []))
        except ValueError:
            sys.stderr.write("reporter:counter:novel,malformed_json_lines,1\n")
            continue
        for genre in genres:
            stdout.write("G:" + genre + "\t1\n")
        for a, b in itertools.combinations(genres, 2):
            stdout.write("P:" + a + "|" + b + "\t1\n")
    stdout.flush()


if __name__ == "__main__":
    main()

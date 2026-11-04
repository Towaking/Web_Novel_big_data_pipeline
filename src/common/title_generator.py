"""
Generator ide judul berbasis pola katalog
Judul web novel sangat formulaik ("Reborn as ...", "... System", "The
Villainess ..."). Modul ini mempelajari pola itu langsung dari judul-judul
katalog lalu mengondisikannya pada cerita pengguna. Bobot tiap judul katalog
berasal dari kemiripan sinopsisnya dengan deskripsi pengguna (ruang embedding
Fase 4), kecocokan genre, dan negara asal (gaya judul CN/KR/JP berbeda jauh).

Dua sumber kandidat:

1. Kerangka katalog (`title_templates.py`, sumber utama): kerangka judul yang
   berulang diisi kata dan frasa cerita yang perannya cocok. Tata bahasanya
   utuh dari judul nyata.
2. Model Markov kata orde-2 (penambal bila kerangka yang cocok kurang):
   sebagian kandidat ditambatkan pada kata cerita, rantai ditarik maju DAN
   mundur dari kemunculan kata itu; kandidat dinilai dari relevansi,
   kelancaran, sambungan terlemah, dan apakah 4-gram di titik sambung pernah
   ada di katalog, karena judul rusak hampir selalu rusak di satu sambungan.

Panjang judul tidak dibatasi, dan judul asal tidak ditampilkan. Tidak ada
model bahasa: kerangka, peran kata, dan trigram semuanya berasal dari katalog.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.title_templates import TemplateTitles

BOS, EOS = "<s>", "</s>"

# Kata yang tetap huruf kecil di tengah judul (gaya judul NovelUpdates).
SMALL_WORDS = {
    "a", "an", "the", "of", "in", "to", "as", "with", "and", "but", "or", "for",
    "on", "at", "by", "from", "into", "is", "vs",
}
# Kata fungsi: tidak dihitung sebagai kata isi saat menilai relevansi.
FUNCTION_WORDS = SMALL_WORDS | {
    "i", "me", "my", "you", "your", "he", "his", "him", "she", "her", "it", "its",
    "we", "our", "they", "their", "this", "that", "who", "was", "be", "am", "are",
    "been", "being", "not", "no", "so", "all", "after", "when", "can", "will",
    "just", "up", "out", "s", "t", "m", "don't", "i'm", "what", "how", "if",
}
DANGLING = SMALL_WORDS | {"i", "my", "your", "his", "her", "their", "our", "its", "who", "that", "when", "after"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[,:!?]")
PUNCT = {",", ":", "!", "?"}
NGRAM_COPY = 5  # frasa sepanjang ini yang hanya dimiliki satu judul = menyalin
#: Jaring pengaman terhadap rantai yang berputar, bukan batas gaya: judul
#: katalog terpanjang 66 token.
MAX_TOKENS = 80


def tokenize_title(title: str) -> list[str]:
    text = str(title).replace("’", "'").replace("‘", "'").replace("–", " - ").replace("—", " - ")
    # Kata bersensor (M*rder) dibuang utuh; memotongnya menghasilkan "M Rder".
    text = re.sub(r"\S*\*\S*", " ", text)
    return TOKEN_RE.findall(text)


def _content(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in FUNCTION_WORDS and t not in PUNCT and len(t) > 1}


@dataclass
class TitleIdea:
    title: str
    relevance: float


class _Chain:
    """Transisi orde-2 terurut per state, untuk sampling berbobot per judul asal."""

    def __init__(self, s1: np.ndarray, s2: np.ndarray, nxt: np.ndarray, title_idx: np.ndarray):
        order = np.lexsort((s2, s1))
        self.s1, self.s2 = s1[order], s2[order]
        self.nxt, self.title_idx = nxt[order], title_idx[order]
        self.starts: dict[tuple, tuple[int, int]] = {}
        for i, key in enumerate(zip(self.s1.tolist(), self.s2.tolist())):
            lo, _ = self.starts.get(key, (i, i))
            self.starts[key] = (lo, i + 1)


class TitleGenerator:
    """Model Markov judul katalog. Dibangun sekali, dikondisikan per permintaan."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        titles = self.catalog["title"].astype(str).tolist()

        surface: dict[str, Counter] = {}
        fwd, bwd = [], []  # (s1, s2, nxt, title_idx) sebagai kata
        ngram_owner: dict[tuple, set] = {}
        self.title_tokens: list[list[str]] = []
        for idx, title in enumerate(titles):
            raw = tokenize_title(title)
            low = [t.lower() for t in raw]
            self.title_tokens.append(low)
            if len(low) < 2:
                continue
            for r, lo in zip(raw, low):
                surface.setdefault(lo, Counter())[r] += 1
            seq = [BOS, BOS] + low + [EOS]
            fwd.extend((seq[i - 2], seq[i - 1], seq[i], idx) for i in range(2, len(seq)))
            rev = [EOS, EOS] + low[::-1] + [BOS]
            bwd.extend((rev[i - 2], rev[i - 1], rev[i], idx) for i in range(2, len(rev)))
            for i in range(len(low) - NGRAM_COPY + 1):
                ngram_owner.setdefault(tuple(low[i:i + NGRAM_COPY]), set()).add(idx)

        # Bentuk tulisan tersering per kata; kata berhuruf besar khusus (CEO,
        # Mr) dipertahankan apa adanya.
        self.surface = {w: c.most_common(1)[0][0] for w, c in surface.items()}
        # Dibandingkan dalam bentuk token, jadi beda tanda kutip/sensor (Sui**de)
        # tidak meloloskan salinan.
        self.title_set = {" ".join(toks) for toks in self.title_tokens}
        # Awal judul yang ada ("The Male Lead" dari "The Male Lead's Fiancee")
        # terbaca sebagai judul terpotong, bukan judul baru.
        self.prefixes = {
            " ".join(toks[:k]) for toks in self.title_tokens for k in range(2, len(toks))
        }
        self.unique_ngrams = {g for g, owners in ngram_owner.items() if len(owners) == 1}
        # 4-gram yang pernah ada (dengan batas awal/akhir judul), penguji
        # sambungan: trigram selalu lolos di titik sambung karena memang begitu
        # judul dirakit, tetapi 4-gram yang melintasinya menunjukkan apakah
        # sambungan itu pernah terjadi di judul asli.
        self.fourgrams = {
            tuple(seq[i:i + 4])
            for toks in self.title_tokens if len(toks) >= 2
            for seq in [[BOS] + toks + [EOS]]
            for i in range(len(seq) - 3)
        }

        self.vocab = sorted({row[2] for row in fwd} | {BOS, EOS})
        self.word_id = {w: i for i, w in enumerate(self.vocab)}
        self._v = len(self.vocab)

        def encode(rows):
            arr = np.array([[self.word_id[a], self.word_id[b], self.word_id[c], t] for a, b, c, t in rows],
                           dtype=np.int64)
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        f1, f2, fn, ft = encode(fwd)
        self.forward = _Chain(f1, f2, fn, ft)
        self.backward = _Chain(*encode(bwd))

        # Model global tanpa bobot (skor kelancaran): trigram + cadangan bigram.
        self._trigram_global = Counter(zip(f1.tolist(), f2.tolist(), fn.tolist()))
        self._state_count = Counter(zip(f1.tolist(), f2.tolist()))
        self._bigram_global = Counter(zip(f2.tolist(), fn.tolist()))
        self._from_count = Counter(f2.tolist())

        # Indeks terbalik kata -> (judul, posisi), untuk menambatkan judul pada
        # kata cerita pengguna.
        self.occurrences: dict[str, list[tuple[int, int]]] = {}
        for idx, toks in enumerate(self.title_tokens):
            if len(toks) < 2:
                continue
            for pos, tok in enumerate(toks):
                if tok not in FUNCTION_WORDS and tok not in PUNCT:
                    self.occurrences.setdefault(tok, []).append((idx, pos))
        self.title_vocab = set(self.vocab) - {BOS, EOS}

        self.frames = TemplateTitles(
            self.title_tokens,
            lambda w: w not in FUNCTION_WORDS and w not in PUNCT and len(w) > 1 and not w.isdigit(),
            BOS, EOS,
        )

    def title_weights(
        self,
        similarity: np.ndarray | None,
        genres: list[str] | None = None,
        origin: str | None = None,
        focus: int = 80,
    ) -> np.ndarray:
        """
        Bobot tiap judul katalog. Semua judul tetap punya bobot dasar supaya
        tata bahasa umum (kata sambung, pola "X of the Y") selalu tersedia;
        `focus` judul termirip mendapat bobot jauh lebih besar.
        """
        n = len(self.catalog)
        weight = np.full(n, 0.05)
        if similarity is not None:
            # NaN = judul yang belum masuk ruang embedding model; hanya bobot dasar.
            similarity = np.nan_to_num(np.asarray(similarity, dtype=float), nan=-1.0)
            top = np.argsort(-similarity)[:focus]
            weight[top] += np.exp(6.0 * np.clip(similarity[top], 0, None))
        if genres:
            wanted = set(genres)
            match = self.catalog["genre_list"].map(
                lambda g: len(wanted.intersection(g if isinstance(g, (list, np.ndarray)) else [])) / len(wanted)
            ).to_numpy(dtype=float)
            weight *= 0.3 + 1.7 * match
        if origin:
            # Kuat: gaya judul JP (kalimat panjang) dan CN (frasa pendek) tidak
            # boleh tercampur hanya karena satu novel CN sangat mirip ceritanya.
            weight *= np.where(self.catalog["type_origin"].to_numpy() == origin, 1.0, 0.05)
        return weight

    def _walk(self, chain: _Chain, state: tuple[int, int], edge_weight, boost, rng, stop: int):
        """Tarik rantai dari `state` sampai simbol `stop`; None bila buntu."""
        tokens, sources = [], []
        s1, s2 = state
        for _ in range(MAX_TOKENS):
            lo, hi = chain.starts.get((s1, s2), (0, 0))
            if hi <= lo:
                return None, None
            w = edge_weight[chain.title_idx[lo:hi]] * boost[chain.nxt[lo:hi]]
            total = w.sum()
            if total <= 0:
                return None, None
            pick = lo + rng.choice(hi - lo, p=w / total)
            word = int(chain.nxt[pick])
            if word == stop:
                return tokens, sources
            tokens.append(self.vocab[word])
            sources.append(int(chain.title_idx[pick]))
            s1, s2 = s2, word
        return None, None

    def _sample_free(self, weights, boost, rng):
        bos = self.word_id[BOS]
        return self._walk(self.forward, (bos, bos), weights, boost, rng, self.word_id[EOS])

    def _sample_anchored(self, word: str, weights, boost, rng):
        """
        Judul yang memuat `word`: ambil satu kemunculannya di judul katalog
        (berbobot), jadikan pasangan dua kata di sekitarnya benih, lalu tarik
        rantai maju sampai akhir dan mundur sampai awal. Setiap trigram tetap
        pernah ada di katalog, termasuk di kedua sisi benih.
        """
        occ = self.occurrences[word]
        p = np.array([weights[t] for t, _ in occ])
        t, pos = occ[rng.choice(len(occ), p=p / p.sum())]
        toks = self.title_tokens[t]
        if pos + 1 < len(toks) and (pos == 0 or rng.random() < 0.5):
            a, b = toks[pos], toks[pos + 1]
        else:
            a, b = toks[pos - 1], toks[pos]
        ia, ib = self.word_id[a], self.word_id[b]
        after, src_a = self._walk(self.forward, (ia, ib), weights, boost, rng, self.word_id[EOS])
        if after is None:
            return None, None
        before, src_b = self._walk(self.backward, (ib, ia), weights, boost, rng, self.word_id[BOS])
        if before is None:
            return None, None
        return before[::-1] + [a, b] + after, src_b[::-1] + [t, t] + src_a

    def _fluency(self, tokens: list[str]) -> tuple[float, float]:
        """
        Log-peluang orde-2 (cadangan orde-1) di model global: rata-rata dan
        sambungan terlemah. Rata-rata menyamarkan satu sambungan rusak di judul
        panjang; nilai minimum menangkapnya.
        """
        ids = [self.word_id[BOS]] * 2 + [self.word_id[t] for t in tokens] + [self.word_id[EOS]]
        logs = []
        for i in range(2, len(ids)):
            tri = self._trigram_global[(ids[i - 2], ids[i - 1], ids[i])]
            st = self._state_count[(ids[i - 2], ids[i - 1])]
            bi = self._bigram_global[(ids[i - 1], ids[i])]
            fr = self._from_count[ids[i - 1]]
            p = 0.7 * (tri / st if st else 0) + 0.3 * ((bi + 0.1) / (fr + 0.1 * self._v))
            logs.append(np.log(p))
        return float(np.mean(logs)), float(np.min(logs))

    def _splices(self, tokens: list[str], sources: list[int]) -> tuple[int, float]:
        """Jumlah titik ganti judul asal, dan porsi yang 4-gram-nya pernah ada di katalog."""
        seq = [BOS] + tokens + [EOS]
        points = [i for i in range(1, len(sources)) if sources[i] != sources[i - 1]]
        if not points:
            return 0, 1.0
        # tokens[i] ada di seq[i + 1]; 4-gram dua kata sebelum dan satu sesudah titik ganti.
        attested = [tuple(seq[i - 1:i + 3]) in self.fourgrams for i in points if i >= 1]
        return len(points), float(np.mean(attested)) if attested else 1.0

    def _is_copy(self, tokens: list[str], sources: list[int]) -> bool:
        # Semua kata dari satu judul = potongan judul itu, bukan judul baru.
        joined = " ".join(tokens)
        if len(set(sources)) < 2 or joined in self.title_set or joined in self.prefixes:
            return True
        for i in range(len(tokens) - NGRAM_COPY + 1):
            if tuple(tokens[i:i + NGRAM_COPY]) in self.unique_ngrams:
                return True
        return False

    def render(self, tokens: list[str]) -> str:
        words = []
        for i, t in enumerate(tokens):
            form = self.surface.get(t, t)
            if t in PUNCT:
                words.append(form)
                continue
            if form == t or form == t.capitalize():
                # Huruf kecil untuk kata sambung di tengah; kapital awal untuk sisanya.
                after_colon = i > 0 and tokens[i - 1] in {":", "!", "?"}
                form = t if (t in SMALL_WORDS and i > 0 and not after_colon) else t[:1].upper() + t[1:]
            words.append(form)
        text = " ".join(words)
        return re.sub(r" ([,:!?])", r"\1", text)

    def anchor_words(self, keywords: set[str], weights: np.ndarray) -> dict[str, float]:
        """
        Kata cerita yang layak jadi tambatan, dengan bobotnya. Kata yang
        dipakai di judul novel mirip ("sect", "villainess") naik; kata prosa
        yang kebetulan pernah muncul di satu-dua judul ("decides") tenggelam.
        """
        base = float(np.mean(weights)) or 1.0
        out: dict[str, float] = {}
        for raw in sorted(keywords):  # urutan tetap: seed yang sama = judul yang sama
            for word in (raw, raw[:-1] if raw.endswith("s") else raw + "s"):
                occ = self.occurrences.get(word)
                if occ:
                    mean_w = float(np.mean([weights[t] for t, _ in occ]))
                    out[word] = max(out.get(word, 0.0), np.log1p(len(occ)) * mean_w / base)
                    break
        return out

    def generate(
        self,
        weights: np.ndarray,
        keywords: set[str],
        n: int = 8,
        seed: int = 0,
        exclude: set[str] | None = None,
        story: str = "",
    ) -> list[TitleIdea]:
        """
        Pilih `n` judul yang beragam. Sumber utama: kerangka katalog yang diisi
        kata cerita (`title_templates`), karena tata bahasanya utuh dari judul
        nyata. Model Markov menambal bila kerangka yang cocok kurang dari `n`.
        `exclude` berisi judul (huruf kecil) yang sudah pernah ditampilkan.
        """
        exclude = exclude or set()
        rng = np.random.default_rng(seed)

        related = [
            w for i in np.argsort(-weights)[:40] for w in self.title_tokens[int(i)]
            if w not in FUNCTION_WORDS and w not in PUNCT
        ]
        fillers = self.frames.story_fillers(story or " ".join(sorted(keywords)), FUNCTION_WORDS,
                                            [w for w, _ in Counter(related).most_common(40)])
        framed = sorted(self.frames.generate(weights, fillers, rng),
                        key=lambda r: r["score"] + rng.normal(0, 0.25), reverse=True)

        chosen: list[dict] = []
        used_frames: set[tuple] = set()

        def offer(tokens: list[str], frame: tuple | None = None) -> bool:
            joined = " ".join(tokens)
            title = self.render(tokens)
            if joined in self.title_set or joined in self.prefixes or title.lower() in exclude:
                return False
            if frame is not None and frame in used_frames:
                return False
            content = _content(tokens)
            if any(len(content & c["content"]) / max(1, len(content | c["content"])) >= 0.5 for c in chosen):
                return False
            chosen.append({"title": title, "content": content})
            if frame is not None:
                used_frames.add(frame)
            return len(chosen) == n

        for row in framed:
            if offer(row["tokens"], row["template"]):
                break
        if len(chosen) < n:
            for tokens in self._markov_candidates(weights, keywords, rng):
                if offer(tokens):
                    break
        return [TitleIdea(title=c["title"], relevance=1.0 - i / max(1, len(chosen)))
                for i, c in enumerate(chosen)]

    def _markov_candidates(self, weights: np.ndarray, keywords: set[str], rng, attempts: int = 900):
        """Kandidat Markov terurut dari skor tertinggi (lihat docstring modul)."""
        anchors = self.anchor_words(keywords, weights)
        anchor_list = list(anchors)
        anchor_p = np.array([anchors[w] for w in anchor_list], dtype=float)
        if anchor_list:
            anchor_p /= anchor_p.sum()
        story_words = set(anchors) | keywords
        boost = np.ones(self._v)
        for word in story_words & self.title_vocab:
            boost[self.word_id[word]] = 4.0

        seen, pool = set(), []
        for attempt in range(attempts):
            if anchor_list and attempt % 2 == 0:
                word = anchor_list[rng.choice(len(anchor_list), p=anchor_p)]
                tokens, sources = self._sample_anchored(word, weights, boost, rng)
            else:
                tokens, sources = self._sample_free(weights, boost, rng)
            if not tokens or len([t for t in tokens if t not in PUNCT]) < 2:
                continue
            first, last = tokens[0], tokens[-1]
            # Judul yang berakhir di kata sambung/kepemilikan ("... with", "... My")
            # atau memuat huruf tunggal sisa singkatan ("Top O") terbaca terpotong.
            if first in PUNCT or last in PUNCT - {"!", "?"} or last in DANGLING:
                continue
            if any(len(t) == 1 and t.isalpha() and t not in {"a", "i"} for t in tokens):
                continue
            words = [t for t in tokens if t not in FUNCTION_WORDS and t not in PUNCT]
            if len(words) != len(set(words)):  # "Sect ... Sect": dua judul tersambung di kata yang sama
                continue
            key = " ".join(tokens)
            if key in seen or self._is_copy(tokens, sources):
                continue
            seen.add(key)
            content = _content(tokens)
            mean_log, min_log = self._fluency(tokens)
            splices, attested = self._splices(tokens, sources)
            pool.append({
                "tokens": tokens, "content": content,
                # Kata isi yang berasal dari cerita pengguna, ditimbang tambatannya.
                "hits": sum(1.0 + min(anchors.get(w, 0.0), 2.0) for w in content & story_words),
                "support": float(np.mean([weights[s] for s in sources])),
                "fluency": mean_log, "weakest": min_log,
                "splices": splices, "attested": attested,
            })
        if not pool:
            return []

        frame = pd.DataFrame(pool)
        # Sambungan yang terlalu jarang di katalog = judul rusak, bukan kreatif.
        floor = frame["weakest"].quantile(0.25)
        frame = frame[frame["weakest"] >= floor].copy()
        z = lambda s: (s - s.mean()) / (s.std() or 1.0)  # noqa: E731
        frame["score"] = (
            1.5 * z(frame["hits"].clip(upper=6))
            + 0.7 * z(np.log(frame["support"]))
            + 0.5 * z(frame["fluency"])
            + 0.6 * z(frame["weakest"])
            + 0.6 * z(frame["attested"])
            # Tiap judul asal tambahan menambah peluang kalimat patah.
            - 0.3 * z(frame["splices"].clip(upper=5))
            # Sedikit acak supaya batch berikutnya tidak dibuka judul yang sama.
            + rng.normal(0, 0.3, len(frame))
        )
        return frame.sort_values("score", ascending=False)["tokens"].tolist()


def description_keywords(text: str) -> set[str]:
    """Kata isi dari deskripsi pengguna, huruf kecil, untuk didorong saat sampling."""
    return _content([t.lower() for t in tokenize_title(text)])

"""
Judul dari kerangka katalog
Model Markov merangkai trigram, jadi sambungan antar-judul sering patah. Modul
ini mengambil jalan lain: kerangka judul yang berulang di katalog ("Return of
the <X>", "I Became the <X>'s <Y>") ditambang, lalu slotnya diisi kata dari
cerita pengguna. Tata bahasanya dipinjam utuh dari judul nyata; hanya isinya
yang baru.

Kata mana yang boleh mengisi slot mana ditentukan oleh **vektor peran** yang
dipelajari dari judul katalog sendiri (distributional semantics): tiap kata
diwakili konteksnya di judul (dua kata kiri dan kanan), matriks PPMI-nya
dipadatkan dengan SVD. Kata yang muncul di posisi serupa berdekatan:
"villainess" dekat "villain", "saint", "heroine"; "hunter" dekat "mage",
"knight". Slot diisi hanya oleh kata yang perannya sedekat pengisi aslinya.

Tidak ada model bahasa atau data eksternal: kerangka, vektor peran, dan bentuk
kata semuanya berasal dari judul katalog.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

SLOT = "<slot>"
#: Kerangka harus dipakai minimal sebanyak ini oleh judul berbeda; di bawahnya
#: "kerangka" hanyalah satu judul dengan lubang.
MIN_SUPPORT = 2
#: Kata dengan frekuensi di bawah ini tidak punya konteks cukup untuk vektor peran.
MIN_WORD_FREQ = 2
ROLE_DIM = 100
#: Huruf yang namanya dibaca berawalan vokal: "an S-Rank", "an F-Class".
VOWEL_LETTERS = set("aefhilmnorsx")
#: Ditandai JJ oleh tagger tetapi bukan pengubah yang bisa berdiri di judul
#: ("his own body" -> "Own Body").
QUANTIFIERS = {"own", "other", "same", "such", "many", "much", "more", "most", "every", "each",
               "several", "various", "certain", "only", "very", "whole", "entire"}


def pos_tagger():
    """
    `nltk.pos_tag_sents` bila data tagger terpasang (image mengunduhnya saat
    build), selain itu None: modul tetap jalan, hanya tanpa pemeriksaan jenis kata.
    """
    try:
        import nltk
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except (ImportError, LookupError):
        return None
    return nltk.pos_tag_sents


def coarse_pos(tag: str) -> str:
    """Penn tag -> kelas kasar: N(oun), J (adjektiva), V(erb), atau O(ther)."""
    if tag.startswith("NN"):
        return "N"
    if tag.startswith("JJ"):
        return "J"
    if tag.startswith("VB"):
        return "V"
    return "O"


def split_possessive(token: str) -> tuple[str, str]:
    return (token[:-2], "'s") if token.endswith("'s") and len(token) > 3 else (token, "")


def form_class(word: str) -> str:
    """Kelas bentuk kata kasar, pengganti POS tagger untuk menjaga kesesuaian."""
    if word.endswith("ing"):
        return "ing"
    if word.endswith("ed"):
        return "ed"
    if word.endswith("ly"):
        return "ly"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return "s"
    return "base"


def fix_articles(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for j in range(len(out) - 1):
        if out[j] in ("a", "an"):
            nxt = out[j + 1]
            if len(nxt) > 1 and nxt[1] == "-":  # S-Rank: dibaca per huruf
                vowel = nxt[0] in VOWEL_LETTERS
            else:
                vowel = nxt[0] in "aeiou"
            out[j] = "an" if vowel else "a"
    return out


class TemplateTitles:
    """Kerangka judul + vektor peran, dibangun sekali dari token judul katalog."""

    def __init__(self, title_tokens: list[list[str]], is_content, begin: str, end: str):
        self.title_tokens = title_tokens
        self.is_content = is_content
        self.tagger = pos_tagger()
        # Jenis kata tiap token judul, dibaca dalam konteks judulnya (~9 detik
        # untuk 39 ribu judul, sekali per versi lake).
        self.title_pos: list[list[str]] | None = None
        if self.tagger is not None:
            self.title_pos = [
                [coarse_pos(tag) for _, tag in tagged]
                for tagged in self.tagger([[split_possessive(t)[0] for t in toks] for toks in title_tokens])
            ]
        self._build_roles(begin, end)
        self._build_templates()

    def _build_roles(self, begin: str, end: str) -> None:
        freq = Counter(split_possessive(t)[0] for toks in self.title_tokens for t in toks)
        self.vocab = [w for w, c in freq.items() if c >= MIN_WORD_FREQ]
        self.word_id = {w: i for i, w in enumerate(self.vocab)}
        contexts: dict[tuple, int] = {}
        rows, cols = [], []
        for toks in self.title_tokens:
            seq = [begin, begin] + [split_possessive(t)[0] for t in toks] + [end, end]
            for i in range(2, len(seq) - 2):
                wid = self.word_id.get(seq[i])
                if wid is None:
                    continue
                for offset in (-2, -1, 1, 2):
                    rows.append(wid)
                    cols.append(contexts.setdefault((offset, seq[i + offset]), len(contexts)))
        counts = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                   shape=(len(self.vocab), len(contexts))).tocsr()
        counts.sum_duplicates()
        counts = counts.tocoo()
        row_sum = np.asarray(counts.sum(1)).ravel()
        # Peluang konteks dihaluskan (pangkat 0,75) supaya konteks langka tidak
        # mendominasi PMI, seperti pada word2vec.
        col_sum = np.asarray(counts.sum(0)).ravel() ** 0.75
        pmi = np.log(counts.data * col_sum.sum() / (row_sum[counts.row] * col_sum[counts.col]))
        keep = pmi > 0
        ppmi = sparse.csr_matrix((pmi[keep], (counts.row[keep], counts.col[keep])), shape=counts.shape)
        roles = TruncatedSVD(ROLE_DIM, random_state=0).fit_transform(ppmi)
        self.roles = roles / (np.linalg.norm(roles, axis=1, keepdims=True) + 1e-9)
        self.bigrams = {
            (split_possessive(a)[0], split_possessive(b)[0])
            for toks in self.title_tokens for a, b in zip(toks, toks[1:])
        }

    def similar_words(self, word: str, k: int = 8) -> list[str]:
        """Kata dengan peran terdekat; untuk inspeksi dan laporan."""
        if word not in self.word_id:
            return []
        sims = self.roles @ self.roles[self.word_id[word]]
        return [self.vocab[i] for i in np.argsort(-sims)[1:k + 1]]

    def _build_templates(self) -> None:
        titles: dict[tuple, set] = defaultdict(set)
        fillers: dict[tuple, Counter] = defaultdict(Counter)
        filler_pos: dict[tuple, Counter] = defaultdict(Counter)
        for idx, toks in enumerate(self.title_tokens):
            if len(toks) < 3:
                continue
            slots = [i for i, t in enumerate(toks) if self.is_content(split_possessive(t)[0])]
            choices = [(i,) for i in slots] + [
                (i, j) for i, j in itertools.combinations(slots, 2) if j > i + 1  # slot bersebelahan = frasa benda acak
            ]
            for positions in choices:
                key = list(toks)
                bases = []
                for p in positions:
                    base, suffix = split_possessive(toks[p])
                    key[p] = SLOT + suffix
                    bases.append(base)
                key = tuple(key)
                titles[key].add(idx)
                for n, base in enumerate(bases):
                    fillers[(key, n)][base] += 1
                    if self.title_pos is not None:
                        filler_pos[(key, n)][self.title_pos[idx][positions[n]]] += 1

        def fixed_content(key):
            return sum(1 for t in key if not t.startswith(SLOT) and self.is_content(split_possessive(t)[0]))

        self.templates: dict[tuple, np.ndarray] = {}
        self.profiles: dict[tuple, tuple] = {}
        for key, owners in titles.items():
            n_slots = sum(t.startswith(SLOT) for t in key)
            # Dua slot butuh minimal satu kata isi tetap, supaya kerangkanya
            # bukan sekadar "<X> of the <Y>" yang bisa jadi apa saja.
            if len(owners) < MIN_SUPPORT or (n_slots == 2 and fixed_content(key) < 1):
                continue
            if sum(1 for t in key if not t.startswith(SLOT)) < 2:
                continue
            profiles = []
            for n in range(n_slots):
                counter = fillers[(key, n)]
                ids = [self.word_id[w] for w in counter if w in self.word_id]
                if len(ids) < 2:
                    break
                weights = np.array([counter[self.vocab[i]] for i in ids], dtype=float)
                centroid = (self.roles[ids] * weights[:, None]).sum(0)
                centroid /= np.linalg.norm(centroid) + 1e-9
                own_fit = self.roles[ids] @ centroid
                forms = Counter(form_class(self.vocab[i]) for i in ids)
                allowed = {f for f, c in forms.items() if c / len(ids) >= 0.2}
                # Jenis kata yang dipakai minimal 20% pengisi asli; None = tanpa tagger.
                tags = filler_pos.get((key, n))
                pos_ok = {p for p, c in tags.items() if c / sum(tags.values()) >= 0.2} if tags else None
                profiles.append((centroid, float(np.quantile(own_fit, 0.25)), self.roles[ids], allowed, pos_ok))
            if len(profiles) == n_slots:
                self.templates[key] = np.fromiter(owners, dtype=np.int64)
                self.profiles[key] = profiles
        self.template_keys = list(self.templates)
        self._support = np.array([len(self.templates[k]) for k in self.template_keys], dtype=float)
        self._fixed = np.array([fixed_content(k) for k in self.template_keys], dtype=float)

    def story_fillers(self, story: str, function_words: set[str], related: list[str]) -> dict[tuple, tuple]:
        """
        Calon pengisi -> (bobot relevansi, jenis kata kasar atau None).

        Dengan tagger: kata benda/sifat/kerja dari cerita, ditambah frasa benda
        dua kata ("sword spirit", "ancestral hall") yang inti katanya dikenal
        judul katalog. Tanpa tagger: frasa hanya bila pernah berdampingan di
        judul katalog, supaya "sword spirit" lolos dan "gates opened" tidak.
        Kata dari judul novel termirip menjadi cadangan berbobot rendah.
        """
        # Tanda baca ikut ditokenisasi: tagger butuh batas kalimatnya, dan frasa
        # tidak boleh melintasinya ("Weak to Strong, Cultivation" bukan
        # "strong cultivation").
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[.,;:!?()]",
                                                story.replace("’", "'"))]
        words = [split_possessive(w)[0] for w in words]
        tags = ([coarse_pos(t) for _, t in self.tagger([words])[0]] if self.tagger and words
                else [None] * len(words))
        tags = ["O" if not w[0].isalnum() else t for w, t in zip(words, tags)]

        def known(w):
            for form in (w, w[:-1] if w.endswith("s") else None):
                if form and form in self.word_id:
                    return form
            return None

        out: dict[tuple, tuple] = {}
        for w, tag in zip(words, tags):
            if w in function_words or len(w) < 3 or tag == "O":
                continue
            form = known(w)
            if form:
                out[(form,)] = (1.0, tag)
        for (a, ta), (b, tb) in zip(zip(words, tags), zip(words[1:], tags[1:])):
            if a in function_words or b in function_words or len(a) < 3 or a in QUANTIFIERS:
                continue
            if not (a[0].isalnum() and b[0].isalnum()):
                continue
            if self.tagger:
                # Pengubah (sifat/benda) + inti benda yang dikenal katalog.
                ok = ta in ("J", "N") and tb == "N" and b in self.word_id
            else:
                ok = (a, b) in self.bigrams and a in self.word_id and b in self.word_id
            if ok:
                out[(a, b)] = (1.3, "N" if self.tagger else None)
        for w in related:
            if w in self.word_id and w not in function_words:
                out.setdefault((w,), (0.35, None))
        return out

    def generate(
        self,
        weights: np.ndarray,
        fillers: dict[tuple, float],
        rng: np.random.Generator,
        top_templates: int = 900,
        per_slot: int = 3,
    ) -> list[dict]:
        """
        Isi kerangka termirip dengan calon pengisi yang perannya cocok. Hasil:
        token judul, skor, dan kerangka asalnya (untuk menjaga keragaman).
        """
        if not fillers or not self.template_keys:
            return []
        phrases = list(fillers)
        relevance = np.array([fillers[p][0] for p in phrases])
        phrase_pos = [fillers[p][1] for p in phrases]
        vectors = self.roles[[self.word_id[p[-1]] for p in phrases]]  # peran frasa = peran kata intinya
        forms = [form_class(p[-1]) for p in phrases]

        base = float(np.mean(weights)) or 1.0
        style = np.array([np.mean(weights[self.templates[k]]) for k in self.template_keys]) / base
        # Kerangka yang dipakai novel mirip, sering dipakai, dan masih membawa
        # kata isi sendiri (lebih spesifik) naik ke atas.
        template_score = np.log(style) + 0.4 * np.log(self._support) + 0.3 * np.minimum(self._fixed, 3)
        order = np.argsort(-(template_score + rng.normal(0, 0.15, len(template_score))))[:top_templates]

        pool = []
        for t in order:
            key = self.template_keys[t]
            slots = [j for j, tok in enumerate(key) if tok.startswith(SLOT)]
            picks = []
            for (centroid, threshold, filler_vecs, allowed, pos_ok) in self.profiles[key]:
                fit = 0.5 * (vectors @ centroid) + 0.5 * (vectors @ filler_vecs.T).max(1)
                ok = [
                    i for i in np.where(fit >= max(threshold, 0.45))[0]
                    if forms[i] in allowed
                    # Kata cerita membawa jenis katanya dari kalimat asalnya;
                    # "holy" (sifat) tidak masuk slot yang selalu diisi benda.
                    and (pos_ok is None or phrase_pos[i] is None or phrase_pos[i] in pos_ok)
                ]
                if not ok:
                    break
                ok = np.array(ok)
                rank = fit[ok] + 0.5 * relevance[ok] + rng.normal(0, 0.05, len(ok))
                picks.append([(phrases[i], fit[i], relevance[i]) for i in ok[np.argsort(-rank)[:per_slot]]])
            if len(picks) != len(slots):
                continue
            for combo in itertools.product(*picks):
                words = [w for phrase, _, _ in combo for w in phrase]
                # Wajib membawa minimal satu kata cerita, dan tak ada kata ganda.
                if len(set(words)) < len(words) or max(c[2] for c in combo) < 1.0:
                    continue
                tokens = list(key)
                for j, (phrase, _, _) in sorted(zip(slots, combo), reverse=True):
                    suffix = tokens[j][len(SLOT):]
                    tokens[j:j + 1] = list(phrase[:-1]) + [phrase[-1] + suffix]
                fit = float(np.mean([c[1] for c in combo]))
                pool.append({
                    "tokens": fix_articles(tokens), "template": key,
                    "score": template_score[t] + 2.0 * fit + 0.8 * sum(c[2] for c in combo),
                })
        return pool

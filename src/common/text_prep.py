"""
Text preprocessing (jalur inferensi + korpus latih Fase 4)
Cermin Python murni dari pipeline teks Spark ML di Fase 2
(`RegexTokenizer` + `StopWordsRemover`), dengan satu tambahan yang disengaja:
pemulihan kata yang disensor NovelUpdates (lihat `uncensor`).

Kenapa ada dua implementasi:
  - Fase 2 memproses 4.461 dokumen sekaligus -> harus terdistribusi (Spark ML).
  - Fase 5 memproses SATU input pengguna per klik -> menyalakan Spark hanya
    untuk itu memakan detik dan sumber daya besar.

Risiko dari dua implementasi adalah *drift*: jika keduanya membersihkan teks
dengan cara berbeda, distribusi fitur saat inferensi tidak sama dengan saat
training dan akurasi turun diam-diam. Karena itu daftar stopword yang persis
dipakai Fase 2 ikut disimpan sebagai artefak model dan dibaca ulang di sini —
bukan diketik ulang. Fase 4 kini juga membangun teks latihnya lewat modul ini.
"""

from __future__ import annotations

import re

#: Sama dengan RegexTokenizer(minTokenLength=3) di Fase 2.
MIN_TOKEN_LENGTH = 3

_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")

#: NovelUpdates mengganti tepat satu huruf per `*` dan ikut menyensor kata yang memuat
#: substring terlarang (gr*pe = grape), jadi kata itu ikut didaftar. Urutan = prioritas.
_CENSOR_TARGETS = """
sex sexy sexes sexual sexually sexuality heterosexual homosexual bisexual asexual intersex
rape raped rapes raping rapist rapists grape grapes drape drapes scrape scraped
slave slaves slaved slaver slavery slaving enslave enslaved enslaves enslavement slaveholder
scum scummy scumbag scumbags trash trashy trashed trashes trashing trashiest
fellatio cunnilingus handjob handjobs blowjob blowjobs intercourse
masturbation masturbate masturbated masturbating
suicide suicides suicidal murder murders murdered murderer murderers murderous murdering
abuse abused abuses abuser abusive abusing stupid stupidity stupidly stupidest
futanari prostitute prostitutes prostitution pervert perverted perverts
fuck fucks fucked fucker fucking motherfucker fap
orgy orgies orgasm orgasms virgin virgins virginity virginia chastity
tentacle tentacles erotic erotica eroge erogenous
bitch bitches bitchy lewd lewdness sucked licked
shit shitty bullshit batshit dogshit loli lolicon lolibaba butt booty hell
horny thorny genital genitals genitalia congenital
cock cocks cocky cockpit cocktail cocktails cockroach cockroaches cockfighting cockfights cockatiel peacock
cuckold cuckolded kill kills killed killer killing dick damn penis semen sperm horsemen
grooming groomed asshole assholes ass boob boobs breast breasts breasted slut sluts
lust lustful lusting bastard bastards adult adults
molest molested molesting molester molestation pussy porn porno pornographic pornography
cum crotch violate violated violation impregnate impregnated impregnating impregnation
vagina vaginal onahole figurine aventurine urine retard retarded erect erection
naked nude nudity grope groped succubus homophobia condom condoms condominium
ejaculation ejaculate whore whores tits incest incestuous creampie nazis anal canal
lesbian lesbians bdsm clitoris climax vibrator drug drugs femdom pregnant plaything libido kinky hitler
pleasure suppressant video daddy dumb legs meat hookups shota whip
""".split()

_TARGETS_BY_LENGTH: dict[int, list[str]] = {}
for _word in _CENSOR_TARGETS:
    _TARGETS_BY_LENGTH.setdefault(len(_word), []).append(_word)

_CENSORED = re.compile(r"[a-z]*\*[a-z*]*")
_RESOLVED: dict[str, str] = {}
#: Memo dibatasi karena proses Streamlit hidup lama dan menerima input bebas.
_RESOLVED_MAX = 20_000


def _resolve_censored(form: str) -> str:
    word = _RESOLVED.get(form)
    if word is not None:
        return word
    word = form.replace("*", "")
    if word:
        for candidate in _TARGETS_BY_LENGTH.get(len(form), ()):
            if all(f == "*" or f == c for f, c in zip(form, candidate)):
                word = candidate
                break
    if len(_RESOLVED) < _RESOLVED_MAX:
        _RESOLVED[form] = word
    return word


def uncensor(text: str) -> str:
    """
    Pulihkan kata yang disensor NovelUpdates pada teks berhuruf kecil.

    Tanpa ini "mu*der" menjadi token "der" dan "r*pe" hilang, sementara pengguna
    dashboard mengetik kata utuh yang tidak pernah dilihat model.
    """
    if not text or "*" not in text:
        return text
    return _CENSORED.sub(lambda m: _resolve_censored(m.group(0)), text)


def normalize(raw: str) -> str:
    """Lowercase, pulihkan sensor, buang karakter non-huruf, rapatkan spasi."""
    if not raw:
        return ""
    lowered = uncensor(raw.lower())
    stripped = _NON_ALPHA.sub(" ", lowered)
    return _MULTI_SPACE.sub(" ", stripped).strip()


def tokenize(raw: str, stopwords: set[str]) -> list[str]:
    """Normalisasi -> pisah spasi -> buang token pendek & stopword."""
    return [
        token
        for token in normalize(raw).split(" ")
        if len(token) >= MIN_TOKEN_LENGTH and token and token not in stopwords
    ]


def to_corpus(raw: str, stopwords: set[str]) -> str:
    """Hasil akhir yang formatnya sama dengan kolom `corpus_clean` di Fase 2."""
    return " ".join(tokenize(raw, stopwords))


def tags_text(tags: str) -> str:
    """Tag dipisah koma -> teks berspasi untuk blok TF-IDF tag (latih & inferensi)."""
    return uncensor((tags or "").lower()).replace(",", " ")


def build_query_text(title: str = "", tags: str = "") -> str:
    """
    Gabungkan judul dan tag persis seperti pembentukan korpus di Fase 2:
    `corpus_raw = title + " " + tags`.
    """
    return f"{title or ''} {tags or ''}".strip()

import re

from phonemizer import phonemize

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

SYMBOLS = [_pad, *list(_punctuation), *list(_letters), *list(_letters_ipa)]


def basic_english_tokenize(text: str) -> list[str]:
    """Splits text into tokens based on words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text)


def ensure_punctuation(text: str) -> str:
    """Adds a comma to the end of the text if no punctuation is present."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?,;:":
        text = text + ","
    return text


def chunk_text(text: str, max_len: int = 400) -> list[str]:
    """Splits long text into manageable chunks at sentence boundaries."""
    sentences = re.split(r"([.!?]+)", text)
    chunks = []

    for i in range(0, len(sentences) - 1, 2):
        full_sentence = (sentences[i] + sentences[i+1]).strip()
        if not full_sentence:
            continue

        if len(full_sentence) <= max_len:
            chunks.append(ensure_punctuation(full_sentence))
        else:
            words = full_sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_len:
                    temp_chunk += (" " if temp_chunk else "") + word
                else:
                    chunks.append(ensure_punctuation(temp_chunk))
                    temp_chunk = word
            if temp_chunk:
                chunks.append(ensure_punctuation(temp_chunk))

    if len(sentences) % 2 != 0 and sentences[-1].strip():
        chunks.append(ensure_punctuation(sentences[-1].strip()))

    return chunks


class Tokenizer:
    def __init__(self, language: str = "en-us"):
        self.symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
        self.id_to_symbol = dict(enumerate(SYMBOLS))
        self.language = language

    def encode(self, text: str, phonemize_text: bool = True) -> list[int]:
        """Converts text to a list of symbol IDs, using a simple phonemize call."""
        if phonemize_text and phonemize:
            # Simple high-level phonemize call
            ps = phonemize(
                text,
                language=self.language,
                strip=True,
                preserve_punctuation=True,
                with_stress=True
            )
            text = " ".join(basic_english_tokenize(ps))
        ids = [self.symbol_to_id[s] for s in text if s in self.symbol_to_id]
        return [0, *ids, 10, 0]
    def decode(self, ids: list[int]) -> str:
        """Converts a list of IDs back to a string."""
        return "".join([self.id_to_symbol[i] for i in ids if i in self.id_to_symbol])

    def __len__(self):
        return len(SYMBOLS)


class TextCleaner:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

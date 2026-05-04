# Symbols and Text Cleaning for StyleTTS2

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
SYMBOLS = [_pad, *list(_punctuation), *list(_letters), *list(_letters_ipa)]


class Tokenizer:
    def __init__(self):
        self.symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
        self.id_to_symbol = dict(enumerate(SYMBOLS))

    def encode(self, text: str) -> list[int]:
        # Simple character-to-id mapping for StyleTTS2
        return [self.symbol_to_id[s] for s in text if s in self.symbol_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.id_to_symbol[i] for i in ids if i in self.id_to_symbol])

    def __len__(self):
        return len(SYMBOLS)


class TextCleaner:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def __call__(self, text: str) -> list[int]:
        # Input text is assumed to be already phonemized (IPA format).
        # We perform minimal cleaning to preserve phoneme integrity.
        # 1. Strip whitespace
        # 2. Convert to tokens using the tokenizer
        text = text.strip()
        # Note: Lowercasing is usually safe for IPA but we do it just in case
        # the phonemizer outputs characters that match our letters list.
        text = text.lower()
        return self.tokenizer.encode(text)

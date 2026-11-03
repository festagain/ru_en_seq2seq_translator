
import pandas as pd
from src.preprocessing import preprocess_parallel_data
from sklearn.model_selection import train_test_split
from pathlib import Path
import sentencepiece as spm
SEED = 39
dataset = pd.read_table("data/dataset_small.tsv", quoting=3, names=["ru_id", "russian", "en_id", "english"])
data, _ = preprocess_parallel_data(dataset)
train, temp = train_test_split(data, train_size = 0.8, random_state = SEED, shuffle = True)
valid, test = train_test_split(temp, train_size = 0.5, random_state = SEED, shuffle = True)
TOKENIZER_DIR = Path("artifacts/tokenizers")
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

ru_train_path = TOKENIZER_DIR / "train_ru.txt"
en_train_path = TOKENIZER_DIR / "train_en.txt"

train["russian"].to_csv(
    ru_train_path,
    index=False,
    header=False
)

train["english"].to_csv(
    en_train_path,
    index=False,
    header=False
)

spm.SentencePieceTrainer.train(
    input=str(ru_train_path),
    model_prefix=str(TOKENIZER_DIR / "ru_bpe"),
    model_type="bpe",
    vocab_size=8_000,
    character_coverage=1,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<sos>",
    eos_piece="<eos>",
    shuffle_input_sentence=False,
    num_threads=1
)

spm.SentencePieceTrainer.train(
    input=str(en_train_path),
    model_prefix=str(TOKENIZER_DIR / "en_bpe"),
    model_type="bpe",
    vocab_size=8_000,
    character_coverage=1,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<sos>",
    eos_piece="<eos>"
)
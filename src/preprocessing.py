import re
import unicodedata


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_parallel_data(data, min_words=1, max_words=30):
    data = data.copy()

    data["russian"] = data["russian"].map(clean_text)
    data["english"] = data["english"].map(clean_text)

    data = data[
        (data["russian"] != "") &
        (data["english"] != "")
    ].copy()

    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=["russian", "english"]
    ).copy()

    removed_duplicates = before_duplicates - len(data)

    data["ru_length"] = data["russian"].str.split().str.len()
    data["en_length"] = data["english"].str.split().str.len()

    mask = (
        data["ru_length"].between(min_words, max_words) &
        data["en_length"].between(min_words, max_words)
    )

    removed_by_length = (~mask).sum()

    data = data.loc[mask].drop(
        columns=["ru_length", "en_length"]
    ).reset_index(drop=True)

    stats = {
        "removed_duplicates": removed_duplicates,
        "removed_by_length": int(removed_by_length),
        "final_size": len(data)
    }

    return data, stats
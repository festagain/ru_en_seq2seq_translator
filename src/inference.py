import sentencepiece as spm
import torch

from .generation import translate_batch
from .model_baseline import Encoder, Decoder, Seq2Seq
from .model_attention import AttentionDecoder, Seq2SeqAttention


def load_model(
    name,
    checkpoint_path,
    ru_model_path=None,
    en_model_path=None,
    device="cpu"
):
    if name not in {"word_lstm", "bpe_lstm", "bpe_attention_lstm"}:
        raise ValueError(f"Неизвестная модель: {name}")

    is_word = name == "word_lstm"
    is_attention = name == "bpe_attention_lstm"

    if not is_word and (ru_model_path is None or en_model_path is None):
        raise ValueError("Для BPE-модели укажите пути к обоим токенизаторам.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    ru_sp = en_sp = None
    ru_vocab = en_vocab = None

    if is_word:
        ru_vocab = checkpoint["ru_vocab"]
        en_vocab = checkpoint["en_vocab"]

        config = {
            "input_dim": len(ru_vocab),
            "output_dim": len(en_vocab),
            "embedding_dim": checkpoint["embedding_dim"],
            "hidden_dim": checkpoint["hidden_dim"],
            "num_layers": checkpoint["num_layers"],
            "dropout": 0.2,
        }

        source_pad = ru_vocab["<pad>"]
        target_pad = en_vocab["<pad>"]
        bos_id = en_vocab["<sos>"]
        eos_id = en_vocab["<eos>"]
    else:
        config = checkpoint["model_config"]
        ru_sp = spm.SentencePieceProcessor(model_file=str(ru_model_path))
        en_sp = spm.SentencePieceProcessor(model_file=str(en_model_path))

        if config["input_dim"] != ru_sp.get_piece_size():
            raise ValueError("Размер русского словаря не совпадает с checkpoint.")
        if config["output_dim"] != en_sp.get_piece_size():
            raise ValueError("Размер английского словаря не совпадает с checkpoint.")

        source_pad = ru_sp.pad_id()
        target_pad = en_sp.pad_id()
        bos_id = en_sp.bos_id()
        eos_id = en_sp.eos_id()

    encoder = Encoder(
        input_dim=config["input_dim"],
        embed_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_idx=source_pad,
        dropout=config.get("dropout", 0.2)
    )

    decoder_config = {
        "output_dim": config["output_dim"],
        "embed_dim": config["embedding_dim"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "pad_idx": target_pad,
        "dropout": config.get("dropout", 0.2),
    }

    if is_attention:
        decoder = AttentionDecoder(
            **decoder_config,
            attention_dim=config.get("attention_dim", config["hidden_dim"])
        )
        model = Seq2SeqAttention(encoder, decoder)
    else:
        decoder = Decoder(**decoder_config)
        model = Seq2Seq(encoder, decoder)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    best_val_loss = checkpoint.get("best_val_loss")
    del checkpoint

    model.to(device)
    model.eval()

    return {
        "model": model,
        "is_word": is_word,
        "is_attention": is_attention,
        "ru_vocab": ru_vocab,
        "id_to_en": {
            token_id: token for token, token_id in en_vocab.items()
        } if is_word else None,
        "ru_sp": ru_sp,
        "en_sp": en_sp,
        "source_pad": source_pad,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "best_val_loss": best_val_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def translate_sentence(sentence, bundle, max_length=50):
    translations, _ = translate_batch([sentence], bundle, max_length=max_length)
    return translations[0]

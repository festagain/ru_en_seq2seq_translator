import re

import torch
from torch.nn.utils.rnn import pad_sequence

from .preprocessing import clean_text


def detokenize_english(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s*-\s*", "-", text)

    return text


@torch.inference_mode()
def translate_batch(sentences, bundle, max_length=50):
    if max_length < 1:
        raise ValueError("max_length должен быть больше нуля.")

    if not sentences:
        return [], 0

    model = bundle["model"]
    model.eval()
    device = next(model.parameters()).device

    encoded = []

    for sentence in sentences:
        sentence = clean_text(sentence)

        if not sentence:
            raise ValueError("Введите непустое предложение для перевода.")

        if bundle["is_word"]:
            vocab = bundle["ru_vocab"]
            tokens = re.findall(r"\w+|[^\w\s]", sentence)
            source_ids = [
                vocab.get(token, vocab["<unk>"])
                for token in tokens
            ]
        else:
            source_ids = bundle["ru_sp"].encode(sentence, out_type=int)

        if not source_ids:
            raise ValueError("После токенизации получилась пустая последовательность.")

        encoded.append(torch.tensor(source_ids, dtype=torch.long))

    source_lengths = torch.tensor(
        [len(source_ids) for source_ids in encoded],
        dtype=torch.long
    )

    sources = pad_sequence(
        encoded,
        batch_first=True,
        padding_value=bundle["source_pad"]
    ).to(device)

    encoder_outputs, hidden, cell = model.encoder(sources, source_lengths)

    source_positions = torch.arange(
        sources.size(1), device=device
    ).unsqueeze(0)
    source_mask = source_positions < source_lengths.to(device).unsqueeze(1)

    input_token = torch.full(
        (len(sentences),),
        bundle["bos_id"],
        dtype=torch.long,
        device=device
    )

    finished = torch.zeros(len(sentences), dtype=torch.bool, device=device)
    generated = []

    for _ in range(max_length):
        if bundle["is_attention"]:
            prediction, hidden, cell, _ = model.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                source_mask
            )
        else:
            prediction, hidden, cell = model.decoder(input_token, hidden, cell)

        next_token = prediction.argmax(dim=1)
        next_token = torch.where(finished, bundle["eos_id"], next_token)
        generated.append(next_token)

        finished |= next_token.eq(bundle["eos_id"])
        input_token = next_token

        if finished.all().item():
            break

    sequences = torch.stack(generated, dim=1).cpu().tolist()
    translations = []

    for token_ids in sequences:
        if bundle["eos_id"] in token_ids:
            token_ids = token_ids[:token_ids.index(bundle["eos_id"])]

        if bundle["is_word"]:
            tokens = [bundle["id_to_en"][token_id] for token_id in token_ids]
            translation = detokenize_english(tokens)
        else:
            translation = bundle["en_sp"].decode(token_ids)

        translations.append(translation)

    without_eos = int((~finished).sum().item())
    return translations, without_eos

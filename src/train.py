import torch
def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    target_pad_idx,
    clip=1.0
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for sources, source_lengths, targets in loader:
        sources = sources.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(
            sources,
            source_lengths,
            targets,
            teacher_forcing_rate=0.5
        )

        output_dim = outputs.size(-1)

        predictions = outputs[:, 1:, :].reshape(-1, output_dim)
        true_tokens = targets[:, 1:].reshape(-1)

        loss = criterion(predictions, true_tokens)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip
        )

        optimizer.step()

        valid_tokens = (
            true_tokens != target_pad_idx
        ).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / total_tokens

def validate_epoch(
    model,
    loader,
    criterion,
    device,
    target_pad_idx,
    teacher_forcing_rate=1.0
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sources, source_lengths, targets in loader:
            sources = sources.to(device)
            targets = targets.to(device)

            outputs = model(
                sources,
                source_lengths,
                targets,
                teacher_forcing_rate=teacher_forcing_rate
            )

            output_dim = outputs.size(-1)

            predictions = outputs[:, 1:, :].reshape(
                -1,
                output_dim
            )

            true_tokens = targets[:, 1:].reshape(-1)

            loss = criterion(predictions, true_tokens)

            valid_tokens = (
                true_tokens != target_pad_idx
            ).sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / total_tokens
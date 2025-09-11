import torch

from metrics import compute_metrics
from utils import save_eval_images, save_sample_images
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    loss_fn,
    print_freq=10,
    log_dir="logs",
    log_file="training.log",
):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir, file_name=log_file)
    header = f"Train: [{epoch}]"
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        inputs = batch["inputs"].to(args.device)
        targets = batch["targets"].to(args.device)

        # Forward pass
        pred_l = model(inputs)

        # Compute loss
        loss_dict = loss_fn(pred_l, targets)
        total_loss = loss_dict["total"]

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update metrics
        # Learning rate (for pretty logging and to avoid empty meter on first step)
        if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
            metric_logger.update(lr=float(optimizer.param_groups[0]["lr"]))
        for loss_name, loss_value in loss_dict.items():
            metric_logger.update(**{f"{loss_name}_loss": loss_value.item()})

        # Save sample images
        if batch_idx % (print_freq * 5) == 0:
            save_sample_images(
                inputs, pred_l, targets, batch_idx, epoch, args.output_dir
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    data_loader,
    model,
    epoch,
    loss_fn,
    print_freq=100,
    results_path=None,
    log_dir="logs",
):
    """Evaluate model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Test: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            inputs = batch["inputs"].to(args.device)
            targets = batch["targets"].to(args.device)
            filenames = batch["filenames"]

            # Forward pass
            pred_l = model(inputs)
            # Clamp predictions for fair metrics and saving
            pred_l = pred_l.clamp(0.0, 1.0)

            # Compute loss
            loss_dict = loss_fn(pred_l, targets)
            for loss_name, loss_value in loss_dict.items():
                metric_logger.update(**{f"{loss_name}_loss": loss_value.item()})

            metrics = compute_metrics(targets, pred_l, args.device)

            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

            if args.save_images:
                save_eval_images(
                    inputs, pred_l, targets, filenames, epoch, args.output_dir
                )

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

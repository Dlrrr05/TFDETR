"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


__all__ = ["profile_stats"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], percentile: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _measure_latency(
    model: nn.Module,
    data: Tensor,
    warmup: int,
    iterations: int,
) -> List[float]:
    if warmup < 0:
        raise ValueError("warmup must be greater than or equal to 0")
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    device = data.device
    with torch.inference_mode():
        for _ in range(warmup):
            model(data)
        _synchronize(device)

        if device.type == "cuda":
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for start_event, end_event in zip(start_events, end_events):
                start_event.record()
                model(data)
                end_event.record()
            _synchronize(device)
            return [
                start_event.elapsed_time(end_event)
                for start_event, end_event in zip(start_events, end_events)
            ]

        latency_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(data)
            latency_ms.append((time.perf_counter() - start) * 1000.0)
        return latency_ms


def _profile_flops(model: nn.Module, data: Tensor) -> Tuple[float, str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if data.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"

    _synchronize(data.device)
    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        with_flops=True,
    ) as profiler:
        model(data)
    _synchronize(data.device)

    events = profiler.key_averages()
    flops = sum(float(event.flops or 0) for event in events)
    info = events.table(sort_by=sort_by, row_limit=-1)
    return flops, info


def profile_stats(
    model: nn.Module,
    data: Optional[Tensor] = None,
    shape: Sequence[int] = (1, 3, 640, 640),
    warmup: int = 10,
    iterations: int = 100,
    verbose: bool = False,
    show_profiler_table: bool = False,
) -> Dict[str, Any]:
    is_training = model.training
    model.eval()

    parameter = next(model.parameters())
    device = parameter.device
    if data is None:
        data = torch.rand(*shape, dtype=parameter.dtype, device=device)
    else:
        data = data.to(device=device, dtype=parameter.dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latency_ms = _measure_latency(model, data, warmup=warmup, iterations=iterations)
    num_flops, info = _profile_flops(model, data)
    mean_latency_ms = statistics.fmean(latency_ms)
    throughput = data.shape[0] * 1000.0 / mean_latency_ms
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    stats = {
        "input_shape": list(data.shape),
        "device": str(device),
        "dtype": str(data.dtype),
        "n_parameters": trainable_params,
        "n_parameters_total": total_params,
        "n_parameters_trainable": trainable_params,
        "parameter_size_mb": parameter_bytes / (1024 ** 2),
        "buffer_size_mb": buffer_bytes / (1024 ** 2),
        "n_flops": num_flops,
        "gflops": num_flops / 1e9,
        "latency_ms": {
            "mean": mean_latency_ms,
            "p50": statistics.median(latency_ms),
            "p95": _percentile(latency_ms, 95.0),
            "min": min(latency_ms),
            "max": max(latency_ms),
        },
        "throughput_samples_per_second": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "info": info,
    }

    if show_profiler_table:
        print(info)
    if verbose:
        print("Profile summary")
        print(f"  Input shape: {list(data.shape)}")
        print(f"  Device: {device} ({data.dtype})")
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Parameter size: {stats['parameter_size_mb']:.2f} MiB")
        print(f"  Buffers: {stats['buffer_size_mb']:.2f} MiB")
        print(f"  FLOPs (supported ops): {num_flops:,.0f} ({stats['gflops']:.3f} GFLOPs)")
        print(
            "  Latency: "
            f"{mean_latency_ms:.3f} ms mean, "
            f"{stats['latency_ms']['p50']:.3f} ms p50, "
            f"{stats['latency_ms']['p95']:.3f} ms p95"
        )
        print(f"  Throughput: {throughput:.2f} samples/s")
        if peak_gpu_memory_mb is not None:
            print(f"  Peak GPU memory: {peak_gpu_memory_mb:.2f} MiB")

    if is_training:
        model.train()
    return stats


def _infer_input_shape(cfg: Any, batch_size: int) -> List[int]:
    eval_spatial_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    if len(eval_spatial_size) != 2:
        raise ValueError(f"eval_spatial_size must contain H and W, got {eval_spatial_size}")

    model_name = cfg.yaml_cfg.get("model")
    model_cfg = cfg.yaml_cfg.get(model_name, {})
    backbone_name = model_cfg.get("backbone")
    backbone_cfg = cfg.yaml_cfg.get(backbone_name, {})
    in_chans = int(backbone_cfg.get("in_chans", 3))
    return [batch_size, in_chans, *map(int, eval_spatial_size)]


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        source = "ema.module"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        source = "model"
    else:
        state = checkpoint
        source = "checkpoint"

    model.load_state_dict(state)
    print(f"Loaded {source} weights from {checkpoint_path}")


def _write_json(stats: Dict[str, Any], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in stats.items() if key != "info"}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved profile report to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="checkpoint path")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device; defaults to cuda:0 when available, otherwise cpu",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        help="input shape; defaults to values inferred from the YAML config",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size used when inferring input shape")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup forwards")
    parser.add_argument("--iterations", type=int, default=100, help="number of timed forwards")
    parser.add_argument("--deploy", action="store_true", help="profile model.deploy() instead of model.eval()")
    parser.add_argument("--show-profiler-table", action="store_true", help="print operator-level profiler table")
    parser.add_argument("--output", type=str, help="optional JSON report path")
    parser.add_argument("-u", "--update", nargs="+", help="update YAML config from command line")
    args = parser.parse_args()

    from src.core import YAMLConfig, yaml_utils

    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, **update_dict)
    model = cfg.model
    if args.resume:
        _load_checkpoint(model, args.resume)
    if args.deploy:
        model = model.deploy()
    model = model.to(args.device)

    shape = args.shape or _infer_input_shape(cfg, args.batch_size)
    stats = profile_stats(
        model,
        shape=shape,
        warmup=args.warmup,
        iterations=args.iterations,
        verbose=True,
        show_profiler_table=args.show_profiler_table,
    )
    if args.output:
        _write_json(stats, args.output)

"""Export a pinned Brax PPO checkpoint to a deterministic ONNX gait policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OBSERVATION_SIZE = 103
ACTION_SIZE = 29
POLICY_LAYER_SIZES = (512, 256, 128, 58)
PARITY_LIMIT = 1e-5


def _float_array(value: object, *, name: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_network_config(payload: dict[str, Any]) -> None:
    """Reject any checkpoint whose policy architecture differs from MM-01."""
    kwargs = payload.get("network_factory_kwargs")
    observation = payload.get("observation_size")
    if not isinstance(kwargs, dict) or not isinstance(observation, dict):
        raise ValueError("checkpoint network config is incomplete")
    state = observation.get("state")
    state_shape = state.get("shape") if isinstance(state, dict) else None
    expected = {
        "action_size": ACTION_SIZE,
        "normalize_observations": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} differs from the frozen export contract")
    if state_shape != [OBSERVATION_SIZE]:
        raise ValueError("checkpoint policy observation shape is not 103")
    required_kwargs = {
        "activation": "silu",
        "distribution_type": "tanh_normal",
        "policy_hidden_layer_sizes": [512, 256, 128],
        "policy_obs_key": "state",
        "state_dependent_std": False,
    }
    for key, value in required_kwargs.items():
        if kwargs.get(key) != value:
            raise ValueError(f"checkpoint network setting {key} differs from MM-01")


def validate_policy_parameters(
    mean: npt.NDArray[np.float32],
    std: npt.NDArray[np.float32],
    layers: tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], ...],
) -> None:
    if mean.shape != (OBSERVATION_SIZE,) or std.shape != (OBSERVATION_SIZE,):
        raise ValueError("normalizer shape differs from the 103-value observation contract")
    if np.any(std <= 0.0):
        raise ValueError("normalizer standard deviation must be positive")
    input_size = OBSERVATION_SIZE
    if len(layers) != len(POLICY_LAYER_SIZES):
        raise ValueError("policy layer count differs from the frozen architecture")
    for index, ((kernel, bias), output_size) in enumerate(
        zip(layers, POLICY_LAYER_SIZES, strict=True)
    ):
        if kernel.shape != (input_size, output_size) or bias.shape != (output_size,):
            raise ValueError(f"policy hidden_{index} parameter shape differs from MM-01")
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError(f"policy hidden_{index} contains non-finite values")
        input_size = output_size


def build_policy_model(
    mean: npt.NDArray[np.float32],
    std: npt.NDArray[np.float32],
    layers: tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], ...],
) -> onnx.ModelProto:
    """Build the exact normalized SiLU MLP and deterministic tanh action mode."""
    validate_policy_parameters(mean, std, layers)
    nodes = [
        helper.make_node("Sub", ("obs", "normalizer_mean"), ("centered",)),
        helper.make_node("Div", ("centered", "normalizer_std"), ("normalized",)),
    ]
    initializers = [
        numpy_helper.from_array(mean, "normalizer_mean"),
        numpy_helper.from_array(std, "normalizer_std"),
    ]
    current = "normalized"
    for index, (kernel, bias) in enumerate(layers):
        kernel_name = f"hidden_{index}_kernel"
        bias_name = f"hidden_{index}_bias"
        linear_name = f"hidden_{index}_linear"
        initializers.extend(
            (
                numpy_helper.from_array(kernel, kernel_name),
                numpy_helper.from_array(bias, bias_name),
            )
        )
        nodes.extend(
            (
                helper.make_node("MatMul", (current, kernel_name), (f"hidden_{index}_matmul",)),
                helper.make_node(
                    "Add", (f"hidden_{index}_matmul", bias_name), (linear_name,)
                ),
            )
        )
        current = linear_name
        if index < len(layers) - 1:
            sigmoid_name = f"hidden_{index}_sigmoid"
            activated_name = f"hidden_{index}_activated"
            nodes.extend(
                (
                    helper.make_node("Sigmoid", (current,), (sigmoid_name,)),
                    helper.make_node("Mul", (current, sigmoid_name), (activated_name,)),
                )
            )
            current = activated_name

    initializers.extend(
        (
            numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_starts"),
            numpy_helper.from_array(np.array([ACTION_SIZE], dtype=np.int64), "slice_ends"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), "slice_axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_steps"),
        )
    )
    nodes.extend(
        (
            helper.make_node(
                "Slice",
                (current, "slice_starts", "slice_ends", "slice_axes", "slice_steps"),
                ("action_location",),
            ),
            helper.make_node("Tanh", ("action_location",), ("continuous_actions",)),
        )
    )
    graph = helper.make_graph(
        nodes,
        "mm01_frozen_gait_policy",
        (helper.make_tensor_value_info("obs", TensorProto.FLOAT, (None, OBSERVATION_SIZE)),),
        (
            helper.make_tensor_value_info(
                "continuous_actions", TensorProto.FLOAT, (None, ACTION_SIZE)
            ),
        ),
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="muscle-memory",
        opset_imports=(helper.make_opsetid("", 11),),
    )
    model.metadata_props.add(key="controller_rate_hz", value="100")
    model.metadata_props.add(key="observation_size", value=str(OBSERVATION_SIZE))
    model.metadata_props.add(key="action_size", value=str(ACTION_SIZE))
    onnx.checker.check_model(model)
    return model


def numpy_policy(
    observations: npt.NDArray[np.float32],
    mean: npt.NDArray[np.float32],
    std: npt.NDArray[np.float32],
    layers: tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], ...],
) -> npt.NDArray[np.float32]:
    values = (observations - mean) / std
    for index, (kernel, bias) in enumerate(layers):
        values = values @ kernel + bias
        if index < len(layers) - 1:
            values = values / (1.0 + np.exp(-values))
    return np.tanh(values[:, :ACTION_SIZE]).astype(np.float32)


def verify_onnx_parity(
    model_path: Path,
    observations: npt.NDArray[np.float32],
    reference_actions: npt.NDArray[np.float32],
    *,
    limit: float = PARITY_LIMIT,
) -> float:
    session = ort.InferenceSession(model_path.as_posix(), providers=("CPUExecutionProvider",))
    output = session.run(("continuous_actions",), {"obs": observations})[0]
    if output.shape != reference_actions.shape or not np.isfinite(output).all():
        raise ValueError("ONNX policy returned an invalid action tensor")
    maximum_delta = float(np.max(np.abs(output - reference_actions)))
    if not math.isfinite(maximum_delta) or maximum_delta > limit:
        raise ValueError(
            f"ONNX parity delta {maximum_delta:.9g} exceeds limit {limit:.9g}"
        )
    return maximum_delta


def _checkpoint_layers(
    policy_params: dict[str, Any],
) -> tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], ...]:
    params = policy_params.get("params")
    if not isinstance(params, dict):
        raise ValueError("checkpoint policy parameters are missing")
    layers = []
    for index in range(len(POLICY_LAYER_SIZES)):
        layer = params.get(f"hidden_{index}")
        if not isinstance(layer, dict) or set(layer) != {"kernel", "bias"}:
            raise ValueError(f"checkpoint hidden_{index} parameters are incomplete")
        layers.append(
            (
                _float_array(layer["kernel"], name=f"hidden_{index} kernel"),
                _float_array(layer["bias"], name=f"hidden_{index} bias"),
            )
        )
    return tuple(layers)


def latest_checkpoint(run_root: Path) -> Path:
    checkpoints = tuple(
        path
        for path in run_root.rglob("checkpoints/*")
        if path.is_dir() and path.name.isdigit()
    )
    if not checkpoints:
        raise ValueError("training did not produce a numeric checkpoint")
    return max(
        checkpoints,
        key=lambda path: (int(path.name), path.stat().st_mtime_ns, path.as_posix()),
    )


def _resolve_checkpoint(run_root: Path, checkpoint_path: Path | None) -> Path:
    logical_root = run_root.absolute()
    logical_checkpoint = (
        latest_checkpoint(logical_root)
        if checkpoint_path is None
        else checkpoint_path.absolute()
    )
    resolved_root = logical_root.resolve(strict=True)
    resolved = logical_checkpoint.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("checkpoint must be inside the declared training run")
    if not resolved.is_dir() or not resolved.name.isdigit():
        raise ValueError("checkpoint must be a numeric Brax checkpoint directory")
    return logical_root / resolved.relative_to(resolved_root)


def export_checkpoint(
    run_root: Path,
    seed: int,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Export one checkpoint and prove deterministic JAX/ONNX output parity."""
    import jax
    import jax.numpy as jnp
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    run_root = run_root.absolute()
    checkpoint_path = _resolve_checkpoint(run_root, checkpoint_path)
    config_path = checkpoint_path / "ppo_network_config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    validate_network_config(config_payload)
    params = brax_checkpoint.load(checkpoint_path.as_posix())
    mean = _float_array(params[0].mean["state"], name="normalizer mean")
    std = _float_array(params[0].std["state"], name="normalizer std")
    layers = _checkpoint_layers(params[1])
    model = build_policy_model(mean, std, layers)
    model_path = run_root / "controller.onnx"
    onnx.save_model(model, model_path)

    rng = np.random.default_rng(seed)
    observations = np.concatenate(
        (
            np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
            np.ones((1, OBSERVATION_SIZE), dtype=np.float32),
            rng.normal(size=(6, OBSERVATION_SIZE)).astype(np.float32),
        )
    )
    networks = ppo_networks.make_ppo_networks(
        {"state": (OBSERVATION_SIZE,), "privileged_state": (216,)},
        ACTION_SIZE,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_LAYER_SIZES[:-1],
        value_hidden_layer_sizes=POLICY_LAYER_SIZES[:-1],
        activation=jax.nn.silu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="tanh_normal",
        state_dependent_std=False,
    )
    inference = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    observations_by_key = {
        "state": jnp.asarray(observations),
        "privileged_state": jnp.zeros((len(observations), 216), dtype=jnp.float32),
    }
    jax_actions, _ = inference(observations_by_key, jax.random.PRNGKey(seed))
    reference = np.asarray(jax_actions, dtype=np.float32)
    maximum_delta = verify_onnx_parity(model_path, observations, reference)
    evidence = {
        "schema_version": 1,
        "checkpoint": checkpoint_path.relative_to(run_root).as_posix(),
        "sample_count": len(observations),
        "maximum_absolute_delta": maximum_delta,
        "limit": PARITY_LIMIT,
        "passed": True,
    }
    (run_root / "onnx-parity.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            export_checkpoint(args.run_root, args.seed, args.checkpoint),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

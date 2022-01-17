from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from distutils.util import strtobool
import argparse
import optuna
import xprun  # type: ignore
import math
import time
import random
import wandb
from copy import deepcopy
import threading
import numpy as np

from enum import Enum


class SamplingStrategy(Enum):
    OMINUS = 0
    POWER_OF_TWO = 1
    LOGUNIFORM = 2
    UNIFORM = 3


@dataclass
class HyperParam:
    path: str
    sampling_strategy: SamplingStrategy
    min_value: float = -float("inf")
    max_value: float = float("inf")
    constraint: Optional[
        Callable[[Dict[str, float]], Tuple[Optional[float], Optional[float]]]
    ] = None
    transform: Optional[Callable[[Dict[str, float], float], float]] = None

    def suggest(
        self,
        trial: optuna.trial.Trial,
        center: float,
        range: float,
        other_vals: Dict[str, float],
    ) -> Tuple[str, float]:
        min_value, max_value = self.min_value, self.max_value
        if self.constraint is not None:
            _min_value, _max_value = self.constraint(other_vals)
            min_value = max(min_value, _min_value or min_value)
            max_value = min(max_value, _max_value or max_value)
        if self.sampling_strategy == SamplingStrategy.OMINUS:
            min_value = max((1 - center) / range, min_value)
            max_value = min((1 - center) * range, max_value)
            value = 1 - trial.suggest_uniform(self.optuna_name(), min_value, max_value)
        elif self.sampling_strategy == SamplingStrategy.POWER_OF_TWO:
            if center * range < min_value:
                value = min_value
            elif int(center / range) > max_value:
                value = max_value
            else:
                min_value = int(max(center / range, min_value))
                max_value = int(min(center * range, max_value))
                value = 2 ** trial.suggest_int(
                    self.optuna_name(),
                    int(math.log2(min_value)),
                    int(math.log2(max_value)),
                )
        elif self.sampling_strategy == SamplingStrategy.LOGUNIFORM:
            min_value = max(center / range, min_value)
            max_value = min(center * range, max_value)
            value = trial.suggest_loguniform(self.optuna_name(), min_value, max_value)
        elif self.sampling_strategy == SamplingStrategy.UNIFORM:
            min_value = max(center - range, min_value)
            max_value = min(center + range, max_value)
            value = trial.suggest_uniform(f"{self.path}", min_value, max_value)
        if self.transform is not None:
            value = self.transform(other_vals, value)
        return f"{self.path}={value}", value

    def optuna_name(self) -> str:
        if self.sampling_strategy == SamplingStrategy.OMINUS:
            return f"om_{self.path}"
        elif self.sampling_strategy == SamplingStrategy.POWER_OF_TWO:
            return f"lg_{self.path}"
        else:
            return f"{self.path}"


hyper_params = {
    "learning-rate": HyperParam(
        path="learning-rate",
        sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
    "num-envs": HyperParam(
        path="num-envs",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        min_value=2,
    ),
    "processes": HyperParam(
        path="processes",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        constraint=lambda x: (1, x["num-envs"]),
    ),
    "d-model": HyperParam(
        path="d-model",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
    ),
    "n-head": HyperParam(
        path="n-head",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        constraint=lambda x: (1, x.get("d-model")),
    ),
    "d-qk": HyperParam(
        path="d-qk",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
    ),
    "n-layer": HyperParam(
        path="n-layer",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        min_value=1,
    ),
    "num-steps": HyperParam(
        path="num-steps",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
    ),
    "gamma": HyperParam(
        path="gamma",
        sampling_strategy=SamplingStrategy.OMINUS,
    ),
    "minibatch-size": HyperParam(
        path="num-minibatches",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        constraint=lambda x: (16, x["num-envs"] * x["num-steps"]),
        transform=lambda args, val: int(args["num-envs"] * args["num-steps"] // val),
    ),
    "ent-coef": HyperParam(
        path="ent-coef",
        sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
    "vf-coef": HyperParam(
        path="vf-coef",
        sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
    "max-grad-norm": HyperParam(
        path="max-grad-norm",
        sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
}


class HyperOptimizer:
    def __init__(
        self,
        params: List[Tuple[str, float, float]],
        target_metric: str,
        xp_name: str,
        parallelism: int = 6,
        steps: Optional[int] = None,
        time: Optional[int] = None,
        xps_per_trial: int = 1,
        priority: int = 3,
        extra_args: Optional[List[str]] = None,
        average_frac: float = 0.5,
        track: bool = False,
        max_microbatch_size: Optional[int] = None,
        # Only run more than 1 trial if first trial is within variance of best result
        adaptive_trials: bool = False,
    ):
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.time = time
        self.xp_name = xp_name
        xp = xprun.build_xpdef(
            "xprun/train.ron",
            ignore_dirty=False,
            include_dirty=True,
            verbose=False,
        )
        xp.base_name = self.xp_name
        self.extra_args = extra_args
        self.config = xp
        self.study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        self.lock = threading.Lock()
        self.cvar = threading.Condition(self.lock)
        self.running_xps = 0
        self.outstanding_xps = 0
        self.parallelism = parallelism
        self.steps = steps
        self.xps_per_trial = xps_per_trial
        self.trial_results: Dict[int, List[float]] = defaultdict(list)
        self.best_result = None
        self.best_result_se = None
        self.best_config: Optional[str] = None
        self.priority = priority
        self.target_metric = target_metric
        self.average_frac = average_frac
        self.track = track
        self.max_microbatch_size = max_microbatch_size
        self.adaptive_trials = adaptive_trials

        self.last_logged_trial_id = -1
        self.log_in_future: List[Tuple[int, Dict[str, float]]] = []

        self.params = params
        self.steps = steps

    def base_xp_config(self, trial: int) -> Any:
        xp = deepcopy(self.config)
        xp.name = f"{self.xp_name}-{trial}"
        if self.extra_args is not None:
            xp.containers[0].command.extend(self.extra_args)
        xp.containers[0].command.append(f"--total-timesteps={self.steps}")
        if self.time:
            xp.containers[0].command.append(f"--max-train-time={self.time}")
        return xp

    def sample_xp(self, trial: optuna.trial.Trial) -> Tuple[Any, Dict[str, float]]:
        xp = self.base_xp_config(self.trial)
        args: Dict[str, float] = {}
        for path, center, range in self.params:
            arg, value = hyper_params[path].suggest(trial, center, range, args)
            args[path] = value
            xp.containers[0].command.append(f"--{arg}")
        if self.max_microbatch_size is not None:
            if "minibatch-size" in args:
                # minibatch-size is actually transformed value of num-minibatches
                minibatch_size = (
                    args["num-envs"] * args["num-steps"] // args["minibatch-size"]
                )
            else:
                minibatch_size = self.max_microbatch_size

            xp.containers[0].command.append(
                f"--microbatch-size={min(self.max_microbatch_size, minibatch_size)}"
            )
        self.trial += 1
        return xp, args

    def run(self, n_trials: int) -> None:
        default_params: Dict[str, float] = {}
        args: Dict[str, float] = {}
        for name, center, _ in self.params:
            if hyper_params[name].sampling_strategy == SamplingStrategy.POWER_OF_TWO:
                center = int(math.log2(center))
            elif hyper_params[name].sampling_strategy == SamplingStrategy.OMINUS:
                center = 1 - center
            oname = hyper_params[name].optuna_name()
            transform = hyper_params[name].transform
            if transform is not None:
                print(args)
                center = transform(args, center)
            default_params[oname] = center
            args[name] = center
        self.study.enqueue_trial(default_params)

        threads = []
        for trial_id in range(n_trials):
            # Wait until we have a free slot
            with self.lock:
                while self.running_xps >= self.parallelism or self.outstanding_xps > 0:
                    self.cvar.wait()
            trial = self.study.ask()
            xp, args = self.sample_xp(trial)
            self.outstanding_xps += (
                self.xps_per_trial if trial_id == 0 or not self.adaptive_trials else 1
            )
            thread = threading.Thread(
                target=self.run_trial,
                args=(
                    xp,
                    trial,
                    trial_id,
                    args,
                ),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        print(f"Best result: {self.best_result}")
        print(f"Best config: {self.best_config}")

    def run_trial(
        self,
        xp: Any,
        trial: optuna.trial.Trial,
        trial_id: int,
        args: Dict[str, float],
    ) -> None:
        threads = []
        max_remaining_xps = self.xps_per_trial
        next_xps = (
            1 if self.adaptive_trials and not trial_id == 0 else self.xps_per_trial
        )
        xpid = 0
        while max_remaining_xps > 0:
            max_remaining_xps -= next_xps
            for _ in range(next_xps):
                with self.lock:
                    while self.running_xps >= self.parallelism:
                        self.cvar.wait()
                    self.running_xps += 1
                    self.outstanding_xps -= 1
                    _xp = deepcopy(xp)
                    if self.xps_per_trial > 1:
                        _xp.containers[0].command.append(f"--trial={trial_id}")
                        _xp.name = f"{_xp.name}-{xpid}"
                    thread = threading.Thread(
                        target=self.run_xp,
                        args=(
                            _xp,
                            trial_id,
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                    self.cvar.notify()
                xpid += 1
            for thread in threads:
                thread.join()
            with self.lock:
                result = np.array(self.trial_results[trial_id]).mean()
                result_se = (
                    np.array(self.trial_results[trial_id]).std()
                    if len(self.trial_results[trial_id]) > 1
                    else self.best_result_se
                )
                if (
                    self.best_result is not None
                    and result + result_se + self.best_result_se < self.best_result
                ):
                    break
                elif max_remaining_xps > 0:
                    next_xps = min(next_xps * 2, max_remaining_xps)
                    self.outstanding_xps += next_xps
                    print(
                        f"{result} + {result_se} + {self.best_result_se} > {self.best_result}, starting {next_xps} more for trial {trial_id}"
                    )

        with self.lock:
            self.study.tell(trial, result)
            print(f"Trial {trial_id}: {result}")
            if self.track:
                args[self.target_metric] = result
                self.log_in_future.append((trial_id, args))
                # Wandb doesn't like step going backwards
                for _trial_id, _args in sorted(self.log_in_future):
                    if (
                        self.last_logged_trial_id is None
                        or self.last_logged_trial_id + 1 == _trial_id
                    ):
                        wandb.log(
                            _args,
                            step=_trial_id,
                        )
                        self.last_logged_trial_id = _trial_id
            if self.best_result is None or result > self.best_result:
                self.best_result = result
                self.best_result_se = result_se
                command = xp.containers[0].command
                self.best_config = command
                print(f"New best config:\n{' '.join(command)}")

    def run_xp(self, xp: Any, trial: int) -> None:
        for retry in range(10):
            try:
                self.xprun.run(xp, wait=True, priority=self.priority, user="clemens")
            except Exception as e:
                print(f"Failed to run {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue
            break
        else:
            print(f"Failed to run {xp.name}")
            return
        while True:
            try:
                self.xprun.block_until_completed(xp.name)
                break
            except Exception as e:
                print(f"Failed to block_until_completed {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue
        run = list(
            self.wandb.runs("entity-neural-network/enn-ppo", {"config.name": xp.name})
        )[0]
        returns = [
            row[self.target_metric]
            for row in run.scan_history(keys=[self.target_metric])
        ]
        if len(returns) == 0:
            result = -1
        else:
            datapoints = max(1, int(len(returns) * self.average_frac))
            result = np.array(returns[-datapoints:]).mean()
        with self.lock:
            self.running_xps -= 1
            self.trial_results[trial].append(result)
            self.cvar.notify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--params", type=str, nargs="+", default=[])
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--steps", type=float)
    parser.add_argument("--time", type=int)  # max training time in seconds
    parser.add_argument("--xps_per_trial", type=int, default=5)
    parser.add_argument("--priority", type=int, default=3)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-microbatch-size", type=int, default=None)
    parser.add_argument("--adaptive-trials", action="store_true")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--average-frac", type=float, default=0.2
    )  # Datapoints from the last average-frac% steps are used to compute final metric
    parser.add_argument("--target-metric", type=str, default="charts/episodic_return")
    parser.add_argument("nargs", nargs="*")
    args = parser.parse_args()

    run_name = args.run_name or f"optuna-{random.randint(0, 0xffffff):06x}"

    if args.track:
        wandb.init(
            project="enn-ppo-hypertuna",
            entity="entity-neural-network",
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )

    params = []
    for param in args.params:
        path, r = param.split("=")
        center, _range = r.split(":")
        params.append((path, float(center), float(_range)))
    HyperOptimizer(
        params,
        args.target_metric,
        run_name,
        args.parallelism,
        int(args.steps) if args.steps is not None else None,
        args.time,
        xps_per_trial=args.xps_per_trial,
        priority=args.priority,
        extra_args=args.nargs,
        track=args.track,
        max_microbatch_size=args.max_microbatch_size,
        adaptive_trials=args.adaptive_trials,
    ).run(args.n_trials)

"""
poetry run python enn_ppo/enn_ppo/hypertuna.py --track --adaptive-trials --steps=1e9 --time=1200 --n_trials=10 --xps_per_trial=15 --priority=3 --target-metric=charts/episodic_return --parallelism=6 --average-frac=0.05 --max-microbatch-size=4096 \
                                                            --params \
                                                                num-envs=128:8 \
                                                                num-steps=32:8 \
                                                                n-layer=2:2 \
                                                                n-head=2:2 \
                                                                learning-rate=0.005:100 \
                                                                gamma=0.99:10 \
                                                                minibatch-size=8192:8 \
                                                                ent-coef=0.05:100 \
                                                                d-model=32:4 \
                                                             -- --gym-id=MultiSnake --track --env-kwargs='{"num_snakes": 2, "max_snake_length": 11}' --processes=16 --max-grad-norm=10 --anneal-entropy=True --relpos-encoding='{"extent": [10, 10], "position_features": ["x", "y"]}'
"""

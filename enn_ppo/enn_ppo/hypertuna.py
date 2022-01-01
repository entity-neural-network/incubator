from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    constraint: Optional[Callable[[Dict[str, float]], Tuple[float, float]]] = None
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
            min_value = max(min_value, _min_value)
            max_value = min(max_value, _max_value)
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
        parallelism: int = 6,
        steps: Optional[int] = None,
        time: Optional[int] = None,
        xps_per_trial: int = 1,
        priority: int = 3,
        extra_args: Optional[List[str]] = None,
    ):
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.time = time
        self.xp_name = f"optuna-{random.randint(0, 0xffffff):06x}"
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
        self.best_config: Optional[str] = None
        self.priority = priority
        self.target_metric = target_metric

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

    def sample_xp(self, trial: optuna.trial.Trial) -> Any:
        xp = self.base_xp_config(self.trial)
        args: Dict[str, float] = {}
        for path, center, range in self.params:
            arg, value = hyper_params[path].suggest(trial, center, range, args)
            args[path] = value
            xp.containers[0].command.append(f"--{arg}")
        self.trial += 1
        return xp

    def run(self, n_trials: int) -> None:
        default_params = {}
        for name, center, _ in self.params:
            if hyper_params[name].sampling_strategy == SamplingStrategy.POWER_OF_TWO:
                center = int(math.log2(center))
            elif hyper_params[name].sampling_strategy == SamplingStrategy.OMINUS:
                center = 1 - center
            oname = hyper_params[name].optuna_name()
            default_params[oname] = center
        self.study.enqueue_trial(default_params)
        threads = []
        for trial_id in range(n_trials):
            # Wait until we have a free slot
            with self.lock:
                while self.running_xps >= self.parallelism or self.outstanding_xps > 0:
                    self.cvar.wait()
            trial = self.study.ask()
            xp = self.sample_xp(trial)
            self.outstanding_xps += self.xps_per_trial
            thread = threading.Thread(
                target=self.run_trial,
                args=(
                    xp,
                    trial,
                    trial_id,
                ),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        print(f"Best result: {self.best_result}")
        print(f"Best config: {self.best_config}")

    def run_trial(self, xp: Any, trial: optuna.trial.Trial, trial_id: int) -> None:
        threads = []
        for i in range(self.xps_per_trial):
            with self.lock:
                while self.running_xps >= self.parallelism:
                    self.cvar.wait()
                self.running_xps += 1
                self.outstanding_xps -= 1
                _xp = deepcopy(xp)
                if self.xps_per_trial > 1:
                    _xp.containers[0].command.append(f"--trial={trial_id}")
                    _xp.name = f"{_xp.name}-{i}"
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
        for thread in threads:
            thread.join()
        result = np.array(self.trial_results[trial_id]).mean()
        with self.lock:
            self.study.tell(trial, result)
            print(f"Trial {trial_id}: {result}")
            if self.best_result is None or result > self.best_result:
                self.best_result = result
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
            result = np.array(returns[len(returns) // 2 :]).mean()
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
    parser.add_argument("--target-metric", type=str, default="charts/episodic_return")
    parser.add_argument("nargs", nargs="*")
    args = parser.parse_args()
    params = []
    for param in args.params:
        path, r = param.split("=")
        center, _range = r.split(":")
        params.append((path, float(center), float(_range)))
    HyperOptimizer(
        params,
        args.target_metric,
        args.parallelism,
        int(args.steps) if args.steps is not None else None,
        args.time,
        xps_per_trial=args.xps_per_trial,
        priority=args.priority,
        extra_args=args.nargs,
    ).run(args.n_trials)

"""
poetry run python enn_ppo/enn_ppo/hypertuna.py --steps=1e9 --time=60 --n_trials=100 --xps_per_trial=3 \
        --params \
            learning-rate=0.003:100 \
            num-envs=256:4 \
            processes=4:4 \
            d-model=64:16 \
#            d-qk=32:8 \
            n-layer=2:8 \
            num-steps=16:32 \
            gamma=0.99:100 \
            num-minibatches=8:8 \
            ent-coef=0.01:100 \
            vf-coef=0.5:4 \
#            max-grad-norm=0.5:10 \
         -- --gym-id=CherryPick --track --env-kwargs='{"num_cherries": 32}'
#        -- --track --gym-id=MultiSnake '--env-kwargs={"num_snakes": 1, "max_snake_length": 6}'
"""

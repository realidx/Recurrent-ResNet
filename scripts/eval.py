import argparse
import copy
import csv
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import flax.struct
from brax import envs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "third_party" / "scaling-crl"))
sys.path.append(str(REPO_ROOT / "src"))

from evaluator import CrlEvaluator
from scripts import crl_train


@flax.struct.dataclass
class EvalState:
    actor_params: Any
    critic_params: Any
    alpha_params: Any


class _ArgsUnpickler(pickle.Unpickler):
    """Map __main__.Args from training runs to the current Args class."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "Args" and module in {"__main__", "scripts.crl_train", "crl_train"}:
            return crl_train.Args
        return super().find_class(module, name)


def load_args(path: Path) -> Any:
    with path.open("rb") as handle:
        return _ArgsUnpickler(handle).load()


def load_params(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def ensure_defaults(args: Any) -> None:
    defaults = {
        "use_goal_from_obs_tail": False,
        "policy_obs_mode": "env",
        "recur_variant": "plain",
        "recur_stages": 1,
        "recur_step_embed_dim": 0,
        "recur_pre_norm": 0,
        "recur_alpha_mode": "fixed",
        "recur_alpha_init": 1.0,
        "recur_eval_steps": 0,
        "recur_depth_dropout": 0.0,
        "recur_trunc_bptt": 0,
        "recur_init_identity": 1,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def make_env(env_id: str) -> Any:
    if env_id == "reacher":
        from envs.reacher import Reacher

        return Reacher(backend="spring")
    if env_id == "pusher":
        from envs.pusher import Pusher

        return Pusher(backend="spring")
    if env_id == "ant":
        from envs.ant import Ant

        return Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
    if "ant" in env_id and "maze" in env_id:
        if "gen" not in env_id:
            from envs.ant_maze import AntMaze

            return AntMaze(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
                maze_layout_name=env_id[4:],
            )
        from envs.ant_maze_generalization import AntMazeGeneralization

        gen_idx = env_id.find("gen")
        maze_layout_name = env_id[4 : gen_idx - 1]
        generalization_config = env_id[gen_idx + 4 :]
        return AntMazeGeneralization(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
            maze_layout_name=maze_layout_name,
            generalization_config=generalization_config,
        )
    if env_id == "ant_ball":
        from envs.ant_ball import AntBall

        return AntBall(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
    if env_id == "ant_push":
        from envs.ant_push import AntPush

        return AntPush(backend="mjx")
    if env_id == "humanoid":
        from envs.humanoid import Humanoid

        return Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
    if "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze

        return HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
    if env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach

        return ArmReach(backend="mjx")
    if env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy

        return ArmBinpickEasy(backend="mjx")
    if env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard

        return ArmBinpickHard(backend="mjx")
    if env_id == "arm_binpick_easy_EEF":
        from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF

        return ArmBinpickEasyEEF(backend="mjx")
    if "arm_grasp" in env_id:
        from envs.manipulation.arm_grasp import ArmGrasp

        cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
        return ArmGrasp(cube_noise_scale=cube_noise_scale, backend="mjx")
    if env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy

        return ArmPushEasy(backend="mjx")
    if env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard

        return ArmPushHard(backend="mjx")
    if env_id == "simple_maze":
        from envs.simple_maze import SimpleMaze

        return SimpleMaze(backend="spring")
    raise NotImplementedError(f"Unknown env_id: {env_id}")


def set_goal_indices(args: Any, env: Any, goal_dim: int) -> None:
    obs_size = env.observation_size
    if goal_dim <= 0 or goal_dim > obs_size:
        raise ValueError(f"Invalid goal_dim={goal_dim} for obs_size={obs_size}")
    args.goal_dim = goal_dim
    args.goal_end_idx = obs_size
    args.goal_start_idx = obs_size - goal_dim
    args.obs_dim = args.goal_start_idx


def count_params(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def build_policy_obs(args: Any, obs: jnp.ndarray) -> jnp.ndarray:
    squeeze = False
    if obs.ndim == 1:
        obs = obs[None, :]
        squeeze = True
    state = obs[:, : args.obs_dim]
    goal = obs[:, args.goal_start_idx : args.goal_end_idx]
    policy_obs = jnp.concatenate([state, goal], axis=-1)
    return policy_obs[0] if squeeze else policy_obs


def policy_input(args: Any, obs: jnp.ndarray) -> jnp.ndarray:
    if args.policy_obs_mode == "env":
        return obs
    if args.policy_obs_mode == "state_goal":
        return build_policy_obs(args, obs)
    raise ValueError(f"Unknown policy_obs_mode: {args.policy_obs_mode}")


def evaluate_run(
    args: Any,
    params: tuple[Any, Any, Any],
    recur_eval_steps: int | None = None,
    num_eval_envs: int | None = None,
    episode_length: int | None = None,
) -> dict[str, Any]:
    eval_args = copy.deepcopy(args)
    ensure_defaults(eval_args)
    if recur_eval_steps is not None:
        eval_args.recur_eval_steps = recur_eval_steps
    if num_eval_envs is not None:
        eval_args.num_eval_envs = num_eval_envs
    if episode_length is not None:
        eval_args.episode_length = episode_length

    requested_steps = eval_args.recur_eval_steps if eval_args.recur_eval_steps > 0 else eval_args.recur_steps
    eval_recur_steps = requested_steps
    if requested_steps > eval_args.recur_steps:
        can_extend = (
            eval_args.encoder_type in {"recurrent_tied", "recurrent_partial"}
            and eval_args.recur_variant == "plain"
            and getattr(eval_args, "recur_step_embed_dim", 0) == 0
        )
        if can_extend:
            eval_args.recur_steps = requested_steps
            eval_recur_steps = requested_steps
        else:
            eval_recur_steps = eval_args.recur_steps
            print(
                "Requested recur_eval_steps exceeds trained steps; "
                "clamping to trained steps due to step embeddings or untied weights.",
                flush=True,
            )

    if not getattr(eval_args, "eval_env_id", ""):
        eval_args.eval_env_id = eval_args.env_id

    env = make_env(eval_args.env_id)
    env = envs.training.wrap(env, episode_length=eval_args.episode_length)
    if getattr(eval_args, "use_goal_from_obs_tail", False):
        goal_dim = getattr(eval_args, "goal_dim", 0)
        if goal_dim <= 0:
            goal_dim = eval_args.goal_end_idx - eval_args.goal_start_idx
        set_goal_indices(eval_args, env, goal_dim)

    eval_env = make_env(eval_args.eval_env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=eval_args.episode_length)
    if getattr(eval_args, "use_goal_from_obs_tail", False):
        goal_dim = getattr(eval_args, "goal_dim", 0)
        if goal_dim <= 0:
            goal_dim = eval_args.goal_end_idx - eval_args.goal_start_idx
        set_goal_indices(eval_args, eval_env, goal_dim)

    obs_size = env.observation_size
    action_size = env.action_size
    policy_obs_dim = obs_size if eval_args.policy_obs_mode == "env" else eval_args.obs_dim + eval_args.goal_dim

    actor_encoder_type = eval_args.encoder_type if eval_args.recur_apply_to_actor else "untied_resnet"
    critic_encoder_type = eval_args.encoder_type if eval_args.recur_apply_to_critic else "untied_resnet"

    actor = crl_train.Actor(
        action_size=action_size,
        network_width=eval_args.actor_network_width,
        network_depth=eval_args.actor_depth,
        skip_connections=eval_args.actor_skip_connections,
        use_relu=eval_args.use_relu,
        encoder_type=actor_encoder_type,
        recur_steps=eval_args.recur_steps,
        recur_init_identity=eval_args.recur_init_identity,
        recur_trunc_bptt=eval_args.recur_trunc_bptt,
        recur_variant=eval_args.recur_variant,
        recur_stages=eval_args.recur_stages,
        recur_step_embed_dim=eval_args.recur_step_embed_dim,
        recur_pre_norm=eval_args.recur_pre_norm,
        recur_alpha_mode=eval_args.recur_alpha_mode,
        recur_alpha_init=eval_args.recur_alpha_init,
    )

    sa_encoder = crl_train.SA_encoder(
        network_width=eval_args.critic_network_width,
        network_depth=eval_args.critic_depth,
        skip_connections=eval_args.critic_skip_connections,
        use_relu=eval_args.use_relu,
        encoder_type=critic_encoder_type,
        recur_steps=eval_args.recur_steps,
        recur_init_identity=eval_args.recur_init_identity,
        recur_trunc_bptt=eval_args.recur_trunc_bptt,
        recur_variant=eval_args.recur_variant,
        recur_stages=eval_args.recur_stages,
        recur_step_embed_dim=eval_args.recur_step_embed_dim,
        recur_pre_norm=eval_args.recur_pre_norm,
        recur_alpha_mode=eval_args.recur_alpha_mode,
        recur_alpha_init=eval_args.recur_alpha_init,
    )
    g_encoder = crl_train.G_encoder(
        network_width=eval_args.critic_network_width,
        network_depth=eval_args.critic_depth,
        skip_connections=eval_args.critic_skip_connections,
        use_relu=eval_args.use_relu,
        encoder_type=critic_encoder_type,
        recur_steps=eval_args.recur_steps,
        recur_init_identity=eval_args.recur_init_identity,
        recur_trunc_bptt=eval_args.recur_trunc_bptt,
        recur_variant=eval_args.recur_variant,
        recur_stages=eval_args.recur_stages,
        recur_step_embed_dim=eval_args.recur_step_embed_dim,
        recur_pre_norm=eval_args.recur_pre_norm,
        recur_alpha_mode=eval_args.recur_alpha_mode,
        recur_alpha_init=eval_args.recur_alpha_init,
    )

    alpha_params, actor_params, critic_params = params
    training_state = EvalState(
        actor_params=actor_params,
        critic_params=critic_params,
        alpha_params=alpha_params,
    )

    def sample_recur_steps(key: jnp.ndarray) -> int:
        if eval_args.recur_depth_dropout <= 0:
            return int(eval_args.recur_steps)
        min_steps = max(1, int(eval_args.recur_steps * (1 - eval_args.recur_depth_dropout)))
        return int(jax.random.randint(key, (), min_steps, eval_args.recur_steps + 1))

    def fixed_recur_steps() -> int:
        return int(eval_recur_steps)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        actor_steps = fixed_recur_steps()
        policy_obs = policy_input(eval_args, env_state.obs)
        means, _ = actor.apply(training_state.actor_params, policy_obs, steps=actor_steps)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, crl_train.Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        key, step_key, action_key = jax.random.split(key, 3)
        actor_steps = sample_recur_steps(step_key)
        policy_obs = policy_input(eval_args, env_state.obs)
        means, log_stds = actor.apply(training_state.actor_params, policy_obs, steps=actor_steps)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(action_key, shape=means.shape, dtype=means.dtype))
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, crl_train.Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        key, actor_step_key, critic_step_key, action_key = jax.random.split(key, 4)
        actor_steps = sample_recur_steps(actor_step_key)
        critic_steps = sample_recur_steps(critic_step_key)
        keys = jax.random.split(action_key, K)
        policy_obs = policy_input(eval_args, env_state.obs)
        means, log_stds = actor.apply(training_state.actor_params, policy_obs, steps=actor_steps)
        stds = jnp.exp(log_stds)
        actions = jnp.stack(
            [
                nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
                for k in keys
            ]
        )
        state = env_state.obs[:, : eval_args.obs_dim]
        goal = env_state.obs[:, eval_args.goal_start_idx : eval_args.goal_end_idx]

        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(
                training_state.critic_params["sa_encoder"],
                state,
                a,
                steps=critic_steps,
            )
        )(actions)
        g_repr = g_encoder.apply(
            training_state.critic_params["g_encoder"],
            goal,
            steps=critic_steps,
        )
        q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, crl_train.Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    if eval_args.eval_actor == 0:
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=eval_args.num_eval_envs,
            episode_length=eval_args.episode_length,
            key=jax.random.PRNGKey(eval_args.seed),
        )
    elif eval_args.eval_actor == 1:
        eval_actor_key = jax.random.PRNGKey(eval_args.seed + 1)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                extra_fields,
            ),
            eval_env,
            num_eval_envs=eval_args.num_eval_envs,
            episode_length=eval_args.episode_length,
            key=jax.random.PRNGKey(eval_args.seed),
        )
    else:
        eval_actor_key = jax.random.PRNGKey(eval_args.seed + 1)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                eval_args.eval_actor,
                extra_fields,
            ),
            eval_env,
            num_eval_envs=eval_args.num_eval_envs,
            episode_length=eval_args.episode_length,
            key=jax.random.PRNGKey(eval_args.seed),
        )

    metrics = evaluator.run_evaluation(training_state, {})
    metrics["model/param_count"] = count_params(actor_params) + count_params(critic_params) + count_params(alpha_params)
    if eval_args.episode_length > 0:
        if "eval/episode_success" in metrics:
            metrics["eval/success_rate"] = metrics["eval/episode_success"] / eval_args.episode_length
        if "eval/episode_success_easy" in metrics:
            metrics["eval/success_easy_rate"] = metrics["eval/episode_success_easy"] / eval_args.episode_length
        if "eval/episode_success_hard" in metrics:
            metrics["eval/success_hard_rate"] = metrics["eval/episode_success_hard"] / eval_args.episode_length

    metrics["eval/recur_steps"] = eval_recur_steps
    return metrics


def collect_run_dirs(runs: Iterable[Path], runs_dir: Path) -> list[Path]:
    if runs:
        return [path.resolve() for path in runs]
    return sorted({path.parent.resolve() for path in runs_dir.rglob("final.pkl")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRL checkpoints and summarize results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("results_phase1.csv"))
    parser.add_argument("--recur-eval-steps", type=int, nargs="*", default=[])
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dirs = collect_run_dirs(args.runs, args.runs_dir)
    if args.limit and args.limit > 0:
        run_dirs = run_dirs[: args.limit]
    if not run_dirs:
        raise SystemExit("No runs found to evaluate.")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        args_path = run_dir / "args.pkl"
        ckpt_path = run_dir / "final.pkl"
        if not args_path.exists() or not ckpt_path.exists():
            continue
        run_args = load_args(args_path)
        ensure_defaults(run_args)
        params = load_params(ckpt_path)

        eval_steps_list = args.recur_eval_steps or [
            run_args.recur_eval_steps if getattr(run_args, "recur_eval_steps", 0) > 0 else run_args.recur_steps
        ]
        for eval_steps in eval_steps_list:
            if args.dry_run:
                print(f"Would evaluate {run_dir} at eval_steps={eval_steps}")
                continue
            metrics = evaluate_run(
                run_args,
                params,
                recur_eval_steps=eval_steps,
                num_eval_envs=args.num_eval_envs,
                episode_length=args.episode_length,
            )
            row = {
                "run_dir": str(run_dir),
                "exp_name": getattr(run_args, "exp_name", run_dir.name),
                "env_id": run_args.env_id,
                "encoder_type": run_args.encoder_type,
                "recur_steps": run_args.recur_steps,
                "recur_variant": run_args.recur_variant,
                "recur_stages": run_args.recur_stages,
                "recur_eval_steps": eval_steps,
                "actor_depth": run_args.actor_depth,
                "critic_depth": run_args.critic_depth,
                "actor_width": run_args.actor_network_width,
                "critic_width": run_args.critic_network_width,
                "num_envs": run_args.num_envs,
                "num_eval_envs": run_args.num_eval_envs,
                "episode_length": run_args.episode_length,
                "seed": run_args.seed,
            }
            for key, value in metrics.items():
                row[key] = to_float(value) if isinstance(value, (np.ndarray, jnp.ndarray, float, int)) else value
            rows.append(row)

    if args.dry_run:
        return

    if not rows:
        raise SystemExit("No evaluations completed.")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

from .base_atari_env import BaseAtariEnv, base_env_wrapper_fn, parallel_wrapper_fn
import os


def raw_env(**kwargs):
    return BaseAtariEnv(game="wizard_of_wor", num_players=2, mode_num=None, env_name=os.path.basename(__file__)[:-3], **kwargs)


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)

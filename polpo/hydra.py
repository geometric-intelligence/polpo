from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedInterpolationType


def register_dict_lookup_resolver(var_name, dict_):
    OmegaConf.register_new_resolver(var_name, lambda key: dict_[key])


def key_value_instantiate(key, value, key_name=None):
    kwargs = {} if key_name is None else {key_name: key}

    return instantiate(value, **kwargs)


def operate_after_instantiate(value, operation):
    instance = instantiate(value)
    return getattr(instance, operation)()


def instantiate_dict_from_config(cfg, name=None, instantiate_func=None):
    # NB: name controls register
    if instantiate_func is None:
        instantiate_func = key_value_instantiate

    dict_ = {}
    missing = {}
    for key, value in cfg.items():
        try:
            dict_[key] = instantiate_func(key, value)

        except UnsupportedInterpolationType:
            missing[key] = value

    if name is None:
        return dict_

    register_dict_lookup_resolver(name, dict_)
    other = instantiate_dict_from_config(missing, instantiate_func=instantiate_func)

    dict_.update(other)

    return dict_


def load_data(data_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        data_cfg,
        name=name,
        instantiate_func=lambda key, value: operate_after_instantiate(
            value, operation="load"
        ),
    )


def load_models(model_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        model_cfg,
        name=name,
        instantiate_func=lambda key, value: operate_after_instantiate(
            value, operation="create"
        ),
    )

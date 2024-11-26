from hydra.utils import instantiate
from omegaconf import OmegaConf


def register_dict_lookup_resolver(var_name, dict_):
    OmegaConf.register_new_resolver(var_name, lambda key: dict_[key])


def _instantiate(key, value, key_name=None):
    kwargs = {} if key_name is None else {key_name: value}

    return instantiate(value, **kwargs)


def _operate_after_instantiate(value, operation):
    instance = instantiate(value)
    return getattr(instance, operation)()


def _instantiate_dict_from_config(cfg, name=None, instantiate_func=None):
    # NB: name controls register
    if instantiate_func is None:
        instantiate_func = _instantiate

    dict_ = {key: instantiate_func(key, value) for key, value in cfg.items()}

    if name is not None:
        register_dict_lookup_resolver(name, dict_)

    return dict_


def load_variables(variables_cfg, name=None):
    # syntax sugar
    return _instantiate_dict_from_config(
        variables_cfg,
        name=name,
        instantiate_func=lambda key, value: _instantiate(key, value, key_name="id_"),
    )


def load_data(data_cfg, name=None):
    # syntax sugar
    return _instantiate_dict_from_config(
        data_cfg,
        name=name,
        instantiate_func=lambda key, value: _operate_after_instantiate(
            value, operation="load"
        ),
    )


def load_models(model_cfg, name=None):
    # syntax sugar
    return _instantiate_dict_from_config(
        model_cfg,
        name=name,
        instantiate_func=lambda key, value: _operate_after_instantiate(
            value, operation="create"
        ),
    )

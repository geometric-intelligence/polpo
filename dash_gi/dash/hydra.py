from dash_gi.hydra import (
    instantiate_dict_from_config,
    key_value_instantiate,
    operate_after_instantiate,
)


def load_variables(variables_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        variables_cfg,
        name=name,
        instantiate_func=lambda key, value: key_value_instantiate(
            key, value, key_name="id_"
        ),
    )


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

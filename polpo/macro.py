import inspect


def create_to_classes_from_from(module, exceptions=()):
    key = "From"
    for name, obj in inspect.getmembers(module):
        if name in exceptions or key not in name or not inspect.isclass(obj):
            continue

        name_ls = name.split(key)
        if len(name_ls) != 2:
            continue

        prefix, suffix = name_ls
        new_name = f"{suffix}To{prefix}"
        setattr(module, new_name, obj)

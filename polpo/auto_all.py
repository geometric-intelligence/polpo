def auto_all(namespace):
    return [
        name
        for name, obj in namespace.items()
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == namespace["__name__"]
    ]

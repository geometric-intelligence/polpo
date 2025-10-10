import os

import nbformat
import pytest
from geomstats.test.parametrizers import (
    _activate_tests_given_data,
    _collect_all_tests,
    _exec_notebook,
    _raise_missing_testing_data,
    _update_attrs,
)


class NotebooksParametrizer(type):
    def __new__(cls, name, bases, attrs):
        def _create_new_test(path, **kwargs):
            def new_test(self):
                return _exec_notebook(path=path)

            return new_test

        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        paths = testing_data.paths

        for path in paths:
            name = path.split(os.sep)[-1].split(".")[0]

            func_name = f"test_{name}"
            test_func = _create_new_test(path)

            metadata = nbformat.read(path, as_version=4).metadata

            for marker_ in metadata.get("markers", []):
                marker = getattr(pytest.mark, marker_)
                test_func = marker()(test_func)

            attrs[func_name] = test_func

        return super().__new__(cls, name, bases, attrs)


class DataBasedParametrizer(type):
    """Metaclass for test classes driven by data definition.

    It differs from `Parametrizer` because every test data function must have
    an associated test function, instead of the opposite.
    """

    # TODO: bring to geomstats

    def __new__(cls, name, bases, attrs):
        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        test_funcs = _collect_all_tests(attrs, bases, active=False)

        _activate_tests_given_data(test_funcs, testing_data)
        if testing_data.skip_all:
            for test_func in test_funcs.values():
                test_func.add_mark("skip")

        _update_attrs(test_funcs, testing_data, attrs)

        # TODO: improve logic
        testing_data.deactivate = testing_data.skip_all
        if testing_data.deactivate:
            return super().__new__(cls, name, bases, attrs)

        for func in test_funcs.values():
            if func.active and not func.skip:
                break
        else:
            testing_data.deactivate = True

        return super().__new__(cls, name, bases, attrs)


@pytest.fixture(scope="class")
def data_check(request):
    testing_data = request.cls.testing_data
    if testing_data.deactivate:
        pytest.skip()

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    IfCondition,
    IndexMap,
    Map,
    NestingSwapper,
    Pipeline,
    WrapInList,
)


class DictsToXY(Pipeline):
    """
    Convert two key-matching dictionaries into input-output pairs for learning.

    The pipeline expects a list of two dictionaries with identical keys.
    It merges them by key, swaps the nesting to produce (key -> [x, y]) pairs,
    and returns a tuple of two lists: one for inputs (X) and one for targets (y).
    """

    def __init__(self):
        # TODO: can be generalized for more cases
        x_step = IfCondition(
            step=ppdict.DictToValuesList(),
            else_step=WrapInList(),
            condition=lambda x: isinstance(x, dict),
        )

        super().__init__(
            steps=[
                ppdict.DictMerger(),
                NestingSwapper(),
                IndexMap(index=0, step=Map(x_step)),
            ]
        )

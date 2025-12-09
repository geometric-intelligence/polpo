import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import Pipeline


class StageKeysDatasetFilter(Pipeline):
    # TODO: make it week based?
    def __init__(self, df, stage="pre"):
        pre_keys = (
            ppd.DfIsInFilter("stage", [stage])
            + ppd.GroupByColumn("subject", as_dict=True)
            + ppdict.DictMap(ppd.ColumnsSelector("sessionID", as_list=True))
        )(df)

        super().__init__(
            steps=[
                ppdict.DictMap(
                    lambda values, key: ppdict.SelectKeySubset(pre_keys.get(key, {}))(
                        values
                    ),
                    pass_key=True,
                ),
                ppdict.DictFilter(lambda vals: len(vals) > 0),
            ]
        )

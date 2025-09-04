import os

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as pppd
from polpo.preprocessing import (
    BranchingPipeline,
    CachablePipeline,
    Constant,
    ContainsAll,
    EnsureIterable,
    Filter,
    IdentityStep,
    IndexMap,
    Map,
    PartiallyInitializedStep,
    Sorter,
    TupleWith,
)
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.mesh.io import FreeSurferReader, PvReader, PvWriter
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import DigitFinder, StartsWith


def CacheableMeshLoader(
    cache_dir,
    pipe,
    use_cache=True,
    cache=True,
    overwrite=True,
):
    # TODO: move place?
    # TODO: make reader and writer package agnostic?

    # TODO: check DictToValuesList and depth
    cache_pipe = (
        FileFinder(IsFileType(ext="vtk"))
        + Sorter()
        + ppdict.HashWithIncoming(Map(PvReader()))
        + ppdict.DictMap(key_step=lambda x: x.rsplit("/", maxsplit=1)[1].split(".")[0])
        + ppdict.NestDict(sep="-")
        + ppdict.NestedDictMap(ppdict.DictToValuesList(), depth=1)
    )

    # TODO: update depth
    # <struct>-<participant>-<mesh-index>
    to_cache_pipe = IndexMap(
        index=1,
        step=(
            ppdict.NestedDictMap(
                BranchingPipeline([lambda x: range(len(x)), IdentityStep()]) + dict,
                depth=1,
            )
            + ppdict.UnnestDict(sep="-")
        ),
    ) + (
        (lambda data: {f"{data[0]}/{key}": value for key, value in data[1].items()})
        + ppdict.DictToTuplesList()
        + Map(PvWriter(ext="vtk"))
    )

    return CachablePipeline(
        cache_dir,
        pipe,
        cache_pipe,
        to_cache_pipe,
        use_cache=use_cache,
        cache=cache,
        overwrite=overwrite,
    )

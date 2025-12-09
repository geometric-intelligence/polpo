import polpo.lddmm as plddmm

from .base import TemplateFinder


class DeterministicTemplateFinder(TemplateFinder):
    def __init__(self, output_dir, registration_kwargs, prefix="sub-"):
        super().__init__()
        self.output_dir = output_dir
        self.registration_kwargs = registration_kwargs

        self._subj_id2name = lambda subj_id: f"{prefix}{subj_id}"

    def __call__(self, subj_id, subj_dataset):
        output_dir = self.output_dir / self._subj_id2name(subj_id)
        if output_dir.exists():
            return plddmm.io.load_template(output_dir, as_path=True)

        return plddmm.learning.estimate_deterministic_atlas(
            targets=subj_dataset,
            output_dir=output_dir,
            initial_step_size=1e-1,
            **self.registration_kwargs,
        )


class GlobalDeterministicTemplateFinder(DeterministicTemplateFinder):
    # TODO: think about this
    def __init__(self, output_dir, registration_kwargs):
        super().__init__(output_dir, registration_kwargs, prefix="")

    def __call__(self, templates, dataset=None, subj_id="global"):
        return super().__call__(subj_id, templates)

class Protocol:
    # TODO: probably also add preprocessing
    # TODO: add simplified version for Euclidean?
    def __init__(
        self,
        template_finder,
        global_template_finder,
        local_space,
        local_global_space=None,
        global_space=None,
        dataset_filter_for_templates=None,
    ):
        if local_global_space is None:
            local_global_space = local_space

        if global_space is None:
            global_space = local_space

        if dataset_filter_for_templates is None:
            dataset_filter_for_templates = lambda x: x

        self.template_finder = template_finder
        self.global_template_finder = global_template_finder
        self.local_space = local_space
        self.local_global_space = local_global_space
        self.global_space = global_space
        self.dataset_filter_for_templates = dataset_filter_for_templates

        # TODO: think about what to store
        self.filtered_dataset_ = None
        self.templates_ = None
        self.subj_logs_ = None
        self.transp_vecs_ = None
        self.global_reprs_ = None

        self.global_id = "global"

    def run(self, dataset):
        self.filtered_dataset_ = self.dataset_filter_for_templates(dataset)

        self.templates_ = self.find_templates(self.filtered_dataset_)

        self.subj_logs_ = self.compute_subj_logs(self.templates_, dataset)
        self.transp_vecs_ = self.compute_parallel_transport(
            self.templates_, self.subj_logs_
        )
        self.global_reprs_ = self.compute_global_repr(
            self.templates_[self.global_id], self.transp_vecs_
        )

        return self

    def find_templates(self, dataset):
        templates = {}
        for subj_id, subj_dataset in dataset.items():
            templates[subj_id] = self.template_finder(subj_id, subj_dataset)

        templates[self.global_id] = self.global_template_finder(
            templates, subj_id=self.global_id, dataset=dataset
        )

        return templates

    def compute_subj_logs(self, templates, dataset):
        logs = {}
        for subj_id, subj_dataset in dataset.items():
            template = templates[subj_id]
            subj_logs = self.local_space.metric.log(
                list(subj_dataset.values()), template
            )

            logs[subj_id] = dict(zip(subj_dataset.keys(), subj_logs))

        # TODO: option to write?
        return logs

    def compute_parallel_transport(self, templates, logs):
        global_template = templates[self.global_id]
        transported_vecs = {}
        for subj_id, subj_logs in logs.items():
            subj_template = templates[subj_id]

            transport_direction = self.local_global_space.metric.log(
                global_template, subj_template
            )

            subj_transported_vecs = self.local_global_space.metric.parallel_transport(
                list(subj_logs.values()), subj_template, direction=transport_direction
            )
            transported_vecs[subj_id] = dict(
                zip(subj_logs.keys(), subj_transported_vecs)
            )

        # TODO: option to write?
        return transported_vecs

    def compute_global_repr(self, global_template, transported_vecs):
        global_reprs = {}
        for subj_id, subj_transported_vecs in transported_vecs.items():
            subj_reprs = self.global_space.metric.exp(
                list(subj_transported_vecs.values()), global_template
            )
            global_reprs[subj_id] = dict(zip(subj_transported_vecs.keys(), subj_reprs))

        # TODO: option to write?
        return global_reprs

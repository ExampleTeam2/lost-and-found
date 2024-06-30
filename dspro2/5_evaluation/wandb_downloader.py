import wandb


class WandbDownloader:
    def __init__(self, entity, project, data_augmentation=None, datasize=None, input_image_size=None, top_n_runs=5):
        self.entity = entity
        self.project = project
        self.top_n_runs = top_n_runs
        self.data_augmentation = data_augmentation
        self.datasize = datasize
        self.input_image_size = input_image_size
        self.api = wandb.Api()
        wandb.login()

    def _throw_on_none(self, value, additional_message=""):
        if value is None:
            raise ValueError("Value cannot be None" + additional_message)
        else:
            return value

    def get_best_runs(self, metric_name, metric_ascending=False):
        runs = self.api.runs(f"{self.entity}/{self.project}")

        filtered_runs = []
        for run in runs:

            config = run.config
            matches = True
            if self.data_augmentation is not None:
                actual_aug = config.get("data_augmentation")
                matches &= actual_aug == self.data_augmentation
            if self.datasize is not None:
                actual_datasize = config.get("datasize") or config.get("dataset_size")
                matches &= actual_datasize == self.datasize
            if self.input_image_size is not None:
                actual_image_size = config.get("input_image_size")
                matches &= actual_image_size == self.input_image_size

            if matches:
                # Skip if the run is running
                if run.state == "running":
                    print(f"Run {run.id} is still running, skipping")
                    continue
                filtered_runs.append(run)

        # Sort in descending order and take the top_n elements
        best_runs = sorted(filtered_runs, key=lambda run: self._throw_on_none(run.summary.get(metric_name, float("-inf") if not metric_ascending else float("inf")), f", metric {metric_name} of run {run.id}"), reverse=(not metric_ascending))[: self.top_n_runs]
        return best_runs

    @staticmethod
    def get_run_data(api, entity, project, run, file_names):
        run_info = {"id": run.id, "parameters": run.config, "metrics": run.summary, "files": {}}
        for file in run.files():
            if any(file.name.endswith(extension) for extension in file_names):
                key_name = "best_model" if file.name.endswith(".pth") and (not file.name.endswith("test_data.pth")) and "best_model" not in run_info["files"] else file.name.split("/")[-1]
                run_info["files"][key_name] = file.url

        # Check for test_data.pth
        if "test_data_run_id" in run.summary:
            test_data_run_id = run.summary.get("test_data_run_id")
            if test_data_run_id:
                try:
                    test_data_run = api.run(f"{entity}/{project}/{test_data_run_id}")
                    for file in test_data_run.files():
                        if file.name.endswith("test_data.pth"):
                            run_info["files"]["test_data"] = file.url
                            break
                except:
                    print(f"Could not find test_data.pth for run {run.id}, looks like run {test_data_run_id} was deleted or is part of a different project.")
        # find file that ends with test_data.pth
        elif "test_data.pth" in run_info["files"]:
            run_info["files"]["test_data"] = run_info["files"]["test_data.pth"]
        return run_info

    def collect_run_data(self, runs, file_names):
        run_data = {}
        for i, run in enumerate(runs, start=1):
            run_key = f"Best Run {i}"
            run_info = WandbDownloader.get_run_data(self.api, self.entity, self.project, run, file_names)
            run_data[run_key] = run_info
        return run_data

    def get_and_collect_best_runs(self, metric_name, file_names, metric_ascending=False):
        best_runs = self.get_best_runs(metric_name, metric_ascending=metric_ascending)
        if best_runs:
            print(f"{self.project}: Found {len(best_runs)} matching runs for datasize {self.datasize} and {self.data_augmentation}.")
        else:
            print(f"{self.project}: No matching runs found for datasize {self.datasize} and {self.data_augmentation}.")
        run_data = self.collect_run_data(best_runs, file_names)
        return run_data

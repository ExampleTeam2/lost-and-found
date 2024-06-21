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

    def get_best_runs(self, metric_name):
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
                filtered_runs.append(run)
        
        # Sort in descending order and take the top_n elements
        best_runs = sorted(filtered_runs, key=lambda run: run.summary.get(metric_name, float("-inf")), reverse=True)[:self.top_n_runs]
        return best_runs

    def collect_run_data(self, runs, file_names):
        run_data = {}
        for i, run in enumerate(runs, start=1):
            run_key = f"Best Run {i}"
            run_info = {
                "id": run.id,
                "parameters": run.config,
                "metrics": run.summary,
                "files": {}
            }
            for file in run.files():
                if any(file.name.endswith(extension) for extension in file_names):
                    key_name = "best_model" if file.name.endswith(".pth") and "best_model" not in run_info["files"] else file.name
                    run_info["files"][key_name] = file.url

            # Check for test_data.pth
            if "test_data_run_id" in run.summary:
                test_data_run_id = run.summary.get("test_data_run_id")
                if test_data_run_id:
                    test_data_run = self.api.run(f"{self.entity}/{self.project}/{test_data_run_id}")
                    for file in test_data_run.files():
                        if file.name == "test_data.pth":
                            run_info["files"]["test_data"] = file.url
                            break
            elif "test_data.pth" in run_info["files"]:
                run_info["files"]["test_data"] = run_info["files"]["test_data.pth"]
                run_data[run_key] = run_info
        return run_data

    def get_and_collect_best_runs(self, metric_name, file_names):
        best_runs = self.get_best_runs(metric_name)
        if not best_runs:
            print("No matching runs found.")
        run_data = self.collect_run_data(best_runs, file_names)
        return run_data

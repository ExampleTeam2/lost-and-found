import pandas as pd

from wandb_downloader import WandbDownloader


def GET_RUN_CONFIGURATIONS(name, image_size, original_image_size):
    return [
        {"project": f"dspro2-predicting-{name}", "data_augmentation": "base_augmentation", "datasize": 81505, "image_size": image_size},
        {"project": f"dspro2-predicting-{name}", "data_augmentation": "full_augmentation_v2", "datasize": 81505, "image_size": image_size},
        {"project": f"dspro2-predicting-{name}", "data_augmentation": "base_augmentation", "datasize": 332786, "image_size": image_size},
        {"project": f"dspro2-predicting-{name}", "data_augmentation": "full_augmentation_v2", "datasize": 332786, "image_size": image_size},
        {"project": f"dspro2-predicting-{name}", "data_augmentation": "base_augmentation", "datasize": 79000, "image_size": original_image_size},
    ]


class BestRunLoader:
    def __init__(self, entity, metric_name, project_names, file_names_to_download):
        self.entity = entity
        self.metric_name = metric_name
        self.project_names = project_names
        self.file_names_to_download = file_names_to_download
        self.configurations = self.create_configurations()
        self.results = self.load_results()

    def create_configurations(self):
        configurations = {}
        for name in self.project_names:
            original_image_size = [180, 320]
            image_size = [80, 130]
            configurations[name] = GET_RUN_CONFIGURATIONS(name, image_size, original_image_size)
        return configurations

    def load_best_runs(self, project, data_augmentation, datasize, image_size):
        metric_ascending = False
        if project.endswith("coordinates"):
            metric_ascending = True
        downloader = WandbDownloader(self.entity, project, data_augmentation, datasize, image_size)
        return downloader.get_and_collect_best_runs(self.metric_name, self.file_names_to_download, metric_ascending=metric_ascending)

    def load_results(self):
        results = {}
        for configs in self.configurations.values():
            for config in configs:
                key = f"{config['project']}_{config['data_augmentation']}_{config['datasize']}_{tuple(config['image_size'])}"
                results[key] = self.load_best_runs(config["project"], config["data_augmentation"], config["datasize"], config["image_size"])
        return results

    def get_summary_table(self):
        summary = []
        for key, value in self.results.items():
            project, augmentation, size, *image_size = key.split("_")
            summary.append({"Project": project, "Data Augmentation": augmentation, "Data Size": int(size), "Image Size": tuple(map(int, image_size)), "Number of Runs": len(value)})
        df = pd.DataFrame(summary)
        return df

    def display_summary_table(self):
        df = self.get_summary_table()
        print(df)

    def count_runs_per_project(self):
        project_counts = {}
        for key in self.results:
            project = key.split("_")[0]
            if project not in project_counts:
                project_counts[project] = 0
            project_counts[project] += len(self.results[key])
        return project_counts

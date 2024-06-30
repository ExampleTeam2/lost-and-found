import gc
import os
import shutil

import torch
import wandb

from geo_model_inference import GeoModelInference


class GeoModelEvaluator(GeoModelInference):
    def __init__(self, val_dataloader, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False, test_data_path=None):
        super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)
        self.test_data_path = test_data_path
        self.val_dataloader = val_dataloader

    def evaluate(self, model_type, model_path, use_balanced_accuracy=False, second_balanced_on_countries_only=None, accuracy_per_country=False, median_metric=False):
        self.prepare(model_type=model_type, model_path=model_path)

        if self.test_data_path is not None:
            print("Pushing test data to wandb...")

            run_dir = wandb.run.dir
            # Get directory of the run (without /files)
            run_dir = os.path.dirname(run_dir) if run_dir.endswith("files") else run_dir

            # Copy test data to run directory
            wandb_test_data_path = os.path.join(run_dir, "test_data.pth")
            # write json file
            shutil.copy(self.test_data_path, wandb_test_data_path)
            wandb.save(wandb_test_data_path)
            # Only save the test data once
            self.test_data_path = None

            print("Test data pushed to wandb, stopping run.")

            return

        with torch.no_grad():
            val_epoch = self.run_epoch(self.val_dataloader, is_train=False, use_balanced_accuracy=use_balanced_accuracy, balanced_on_countries_only=None, accuracy_per_country=accuracy_per_country, median_metric=median_metric)

        log = {}

        val_loss = val_epoch["avg_loss"]
        log["Best Validation Loss"] = val_loss
        if self.use_coordinates or self.use_regions:
            val_metric = val_epoch["avg_metric"]
            log["Best Validation Distance (km)"] = val_metric
            if median_metric:
                val_median_metric = val_epoch["median_metric"]
                log["Best Validation Median Distance (km)"] = val_median_metric
        if self.use_regions or (not self.use_coordinates):
            val_top1_accuracy, val_top3_accuracy, val_top5_accuracy = val_epoch["top1_accuracy"], val_epoch["top3_accuracy"], val_epoch["top5_accuracy"]
            log["Best Validation Accuracy Top 1"] = val_top1_accuracy
            log["Best Validation Accuracy Top 3"] = val_top3_accuracy
            log["Best Validation Accuracy Top 5"] = val_top5_accuracy
            if use_balanced_accuracy:
                val_top1_balanced_accuracy = val_epoch["top1_balanced_accuracy"]
                log["Best Validation Balanced Accuracy Top 1"] = val_top1_balanced_accuracy
            if accuracy_per_country:
                val_accuracy_per_country = val_epoch["accuracy_per_country"]
                log["Best Validation Accuracy Per Country"] = val_accuracy_per_country
        if self.use_regions:
            val_top1_correct_country, val_top3_correct_country, val_top5_correct_country = val_epoch["top1_correct_country"], val_epoch["top3_correct_country"], val_epoch["top5_correct_country"]
            log["Best Validation Accuracy Top 1 Country"] = val_top1_correct_country
            log["Best Validation Accuracy Top 3 Country"] = val_top3_correct_country
            log["Best Validation Accuracy Top 5 Country"] = val_top5_correct_country
            if use_balanced_accuracy:
                val_top1_balanced_accuracy_country = val_epoch["top1_balanced_accuracy_country"]
                log["Best Validation Balanced Accuracy Top 1 Country"] = val_top1_balanced_accuracy_country

        if (self.use_regions or (not self.use_coordinates)) and use_balanced_accuracy and second_balanced_on_countries_only is not None:
            with torch.no_grad():
                val_epoch_mapped = self.run_epoch(self.val_dataloader, is_train=False, use_balanced_accuracy=use_balanced_accuracy, balanced_on_countries_only=second_balanced_on_countries_only, accuracy_per_country=False, median_metric=False)
            if not self.use_regions:
                val_top1_balanced_accuracy_mapped = val_epoch_mapped["top1_balanced_accuracy"]
                log["Best Validation Balanced Accuracy Top 1 (Mapped)"] = val_top1_balanced_accuracy_mapped
            else:
                val_top1_balanced_accuracy_country_mapped = val_epoch_mapped["top1_balanced_accuracy_country"]
                log["Best Validation Balanced Accuracy Top 1 Country (Mapped)"] = val_top1_balanced_accuracy_country_mapped

        # Log metrics to wandb
        for k, v in log.items():
            wandb.run.summary[k] = v

        # Clean up
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

        # Print results, floats to 4 decimal places
        for k, v in log.items():
            # if it's a dictionary, print each key-value pair on a new line
            if isinstance(v, dict):
                print(f"{k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2:.4f}")
            else:
                print(f"{k}: {v:.4f}")

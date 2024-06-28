import gc

import torch
import wandb

from geo_model_inference import GeoModelInference


class GeoModelEvaluator(GeoModelInference):
    def __init__(self, val_dataloader, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
        super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)
        self.val_dataloader = val_dataloader

    def evaluate(self, use_balanced_accuracy=False, second_balanced_on_countries_only=None, accuracy_per_country=False, median_metric=False):
        with wandb.init(reinit=True) as run:
            print(f"Evaluating for run {run.name} ({run.id})")

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
                print(f"{k}: {v:.4f}")

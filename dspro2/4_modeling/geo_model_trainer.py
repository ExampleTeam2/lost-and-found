import os
import gc
import json
import shutil

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
import uuid

from geo_model_harness import GeoModelHarness


class GeoModelTrainer(GeoModelHarness):
    def __init__(self, datasize, train_dataloader, val_dataloader, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, test_data_path=None, predict_regions=False, run_start_callback=None):
        super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.datasize = datasize
        self.patience = 11
        self.test_data_path = test_data_path
        self.run_start_callback = run_start_callback

    def train(self):
        with wandb.init(reinit=True) as run:
            config = run.config

            if self.run_start_callback is not None:
                self.run_start_callback(config, run)

            self.set_seed(config.seed)

            # Set seeds, configure optimizers, losses, etc.
            best_val_metric = float("inf") if self.use_coordinates else 0
            best_logs = {}
            patience_counter = 0

            # Rename run name and initialize parameters in model name
            model_name = f"model_{config.model_name}_lr_{config.learning_rate}_opt_{config.optimizer}_weightDecay_{config.weight_decay}_imgSize_{config.input_image_size}"
            run_name = model_name + f"_{uuid.uuid4()}"
            wandb.run.name = run_name

            # Initialize model, optimizer and criterion
            self.initialize_model(model_type=config.model_name)

            if "resnet" in self.model_type:
                optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if not n.startswith("fc")], "lr": config.learning_rate * 0.1}, {"params": self.model.fc.parameters(), "lr": config.learning_rate}]
            elif "efficientnet" in self.model_type or "mobilenet" in self.model_type:
                optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if not n.startswith("classifier")], "lr": config.learning_rate * 0.1}, {"params": self.model.classifier.parameters(), "lr": config.learning_rate}]

            self.optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
            scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

            for epoch in range(config.epochs):
                train_epoch = self.run_epoch(self.train_dataloader, is_train=True, optimizer=self.optimizer)
                val_epoch = self.run_epoch(self.val_dataloader, is_train=False)
                train_loss, val_loss = train_epoch["avg_loss"], val_epoch["avg_loss"]
                if self.use_coordinates or self.use_regions:
                    train_metric, val_metric = train_epoch["avg_metric"], val_epoch["avg_metric"]
                if self.use_regions or (not self.use_coordinates):
                    train_top1_accuracy, train_top3_accuracy, train_top5_accuracy = train_epoch["top1_accuracy"], train_epoch["top3_accuracy"], train_epoch["top5_accuracy"]
                    val_top1_accuracy, val_top3_accuracy, val_top5_accuracy = val_epoch["top1_accuracy"], val_epoch["top3_accuracy"], val_epoch["top5_accuracy"]
                if self.use_regions:
                    train_top1_correct_country, train_top3_correct_country, train_top5_correct_country = train_epoch["top1_correct_country"], train_epoch["top3_correct_country"], train_epoch["top5_correct_country"]
                    val_top1_correct_country, val_top3_correct_country, val_top5_correct_country = val_epoch["top1_correct_country"], val_epoch["top3_correct_country"], val_epoch["top5_correct_country"]

                log = {}

                # Log metrics to wandb
                if self.use_coordinates:
                    log = {"Train Loss": train_loss, "Train Distance (km)": train_metric, "Validation Loss": val_loss, "Validation Distance (km)": val_metric}
                elif self.use_regions:
                    log = {"Train Loss": train_loss, "Train Distance (km)": train_metric, "Train Accuracy Top 1": train_top1_accuracy, "Train Accuracy Top 3": train_top3_accuracy, "Train Accuracy Top 5": train_top5_accuracy, "Train Accuracy Top 1 Country": train_top1_correct_country, "Train Accuracy Top 3 Country": train_top3_correct_country, "Train Accuracy Top 5 Country": train_top5_correct_country, "Validation Loss": val_loss, "Validation Distance (km)": val_metric, "Validation Accuracy Top 1": val_top1_accuracy, "Validation Accuracy Top 3": val_top3_accuracy, "Validation Accuracy Top 5": val_top5_accuracy, "Validation Accuracy Top 1 Country": val_top1_correct_country, "Validation Accuracy Top 3 Country": val_top3_correct_country, "Validation Accuracy Top 5 Country": val_top5_correct_country}
                else:
                    log = {"Train Loss": train_loss, "Train Accuracy Top 1": train_top1_accuracy, "Train Accuracy Top 3": train_top3_accuracy, "Train Accuracy Top 5": train_top5_accuracy, "Validation Loss": val_loss, "Validation Accuracy Top 1": val_top1_accuracy, "Validation Accuracy Top 3": val_top3_accuracy, "Validation Accuracy Top 5": val_top5_accuracy}

                wandb.log(log)

                # Even for predicting regions, always use the best model based on validation distance
                if (self.use_coordinates and val_metric < best_val_metric) or ((not self.use_coordinates) and (not self.use_regions) and val_top1_accuracy > best_val_metric) or (self.use_regions and val_top1_correct_country > best_val_metric):
                    best_logs = log

                    if self.use_coordinates:
                        best_val_metric = val_metric
                    elif self.use_regions:
                        best_val_metric = val_top1_correct_country
                    else:
                        best_val_metric = val_top1_accuracy

                    os.makedirs(f"models/datasize_{self.datasize}", exist_ok=True)
                    raw_model_path = f"best_model_checkpoint{model_name}_predict_coordinates_{self.use_coordinates}.pth"
                    model_path = f"models/datasize_{self.datasize}/{raw_model_path}"
                    torch.save(self.model.state_dict(), model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Stopping early after {self.patience} epochs without improvement")
                        break

                # Step the scheduler at the end of the epoch
                scheduler.step()

            # Filter all keys with "Train" from best_logs and prepend "Best" to all others
            best_logs = {("Best " + k): v for k, v in best_logs.items() if not k.lower().startswith("train")}
            # Push best logs to wandb summary only
            for k, v in best_logs.items():
                wandb.run.summary[k] = v
            #wandb.run.summary.update()

            # Load and log the best model to wandb
            self.initialize_model(model_type=config.model_name)
            self.model.load_state_dict(torch.load(model_path))
            run_dir = wandb.run.dir
            # Get directory of the run (without /files)
            run_dir = os.path.dirname(run_dir) if run_dir.endswith("files") else run_dir
            print("Saving artifacts to:", run_dir)
            wandb_model_path = os.path.join(wandb.run.dir, raw_model_path)

            if self.country_to_index is not None:
                # copy to run directory
                wandb_country_to_index_file = os.path.join(run_dir, "country_to_index.json")
                # write json file
                with open(wandb_country_to_index_file, "w") as f:
                    json.dump(self.country_to_index, f)
                # save to wandb
                wandb.save(wandb_country_to_index_file)

            if self.region_to_index is not None:
                # copy to run directory
                wandb_region_to_index_file = os.path.join(run_dir, "region_to_index.json")
                # write json file
                with open(wandb_region_to_index_file, "w") as f:
                    json.dump(self.region_to_index, f)
                # save to wandb
                wandb.save(wandb_region_to_index_file)

            if self.region_index_to_middle_point is not None:
                # copy to run directory
                wandb_region_index_to_middle_point_file = os.path.join(run_dir, "region_index_to_middle_point.json")
                # write json file
                with open(wandb_region_index_to_middle_point_file, "w") as f:
                    json.dump(self.region_index_to_middle_point, f)
                # save to wandb
                wandb.save(wandb_region_index_to_middle_point_file)

            if self.region_index_to_country_index is not None:
                # copy to run directory
                wandb_region_index_to_country_index_file = os.path.join(run_dir, "region_index_to_country_index.json")
                # write json file
                with open(wandb_region_index_to_country_index_file, "w") as f:
                    json.dump(self.region_index_to_country_index, f)
                # save to wandb
                wandb.save(wandb_region_index_to_country_index_file)

            if self.test_data_path is not None:
                # Copy test data to run directory
                wandb_test_data_path = os.path.join(run_dir, "test_data.pth")
                # write json file
                shutil.copy(self.test_data_path, wandb_test_data_path)
                wandb.save(wandb_test_data_path)
                # Only save the test data once
                self.test_data_path = None

            torch.save(self.model.state_dict(), wandb_model_path)
            wandb.save(wandb_model_path)

            # Clean up
            del self.model

            gc.collect()
            torch.cuda.empty_cache()

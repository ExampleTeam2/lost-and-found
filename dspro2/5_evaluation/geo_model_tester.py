import torch

from geo_model_inference import GeoModelInference


class GeoModelTester(GeoModelInference):
    def __init__(self, test_dataloader, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
        super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)
        self.test_dataloader = test_dataloader

    def test(self, model_type, model_path, balanced_on_countries_only=None, accuracy_per_country=False):
        self.prepare(model_type=model_type, model_path=model_path)

        with torch.no_grad():

            test_epoch = self.run_epoch(self.test_dataloader, is_train=False, use_balanced_accuracy=True, balanced_on_countries_only=balanced_on_countries_only, accuracy_per_country=accuracy_per_country, median_metric=True)
            test_loss = test_epoch["avg_loss"]
            if self.use_coordinates or self.use_regions:
                test_metric = test_epoch["avg_metric"]
                test_median_metric = test_epoch["median_metric"]
            if self.use_regions or (not self.use_coordinates):
                test_top1_accuracy, test_top3_accuracy, test_top5_accuracy = test_epoch["top1_accuracy"], test_epoch["top3_accuracy"], test_epoch["top5_accuracy"]
                test_top1_balanced_accuracy = test_epoch["top1_balanced_accuracy"]
                if accuracy_per_country:
                    accuracy_per_country = test_epoch["accuracy_per_country"]
            if self.use_regions:
                test_top1_correct_country, test_top3_correct_country, test_top5_correct_country = test_epoch["top1_correct_country"], test_epoch["top3_correct_country"], test_epoch["top5_correct_country"]
                test_top1_balanced_accuracy_country = test_epoch["top1_balanced_accuracy_country"]

        print(f"Network-Model: {model_type}")
        parts = model_path.split('/')
        project_name = parts[5].rsplit('-', 1)[-1]
        run_id = parts[6]
        print(f"Project Name: {project_name}")
        print(f"Run ID: {run_id}")
        if self.use_coordinates:
            print(f"Test Loss: {test_loss:.4f}, Test Distance: {test_metric:.4f}, Test Distance Median: {test_median_metric:.4f}")
        elif self.use_regions:
            print(f"Test Loss: {test_loss:.4f}, Test Distance: {test_metric:.4f}, Test Distance Median: {test_median_metric:.4f}")
            print(f"Test Top 1 Accuracy: {test_top1_accuracy:.4f}, Test Top 3 Accuracy: {test_top3_accuracy:.4f}, Test Top 5 Accuracy: {test_top5_accuracy:.4f}")
            print(f"Test Top 1 Accuracy (Country): {test_top1_correct_country:.4f}, Test Top 3 Accuracy (Country): {test_top3_correct_country:.4f}, Test Top 5 Accuracy (Country): {test_top5_correct_country:.4f}")
            print(f"Test Top 1 Balanced Accuracy: {test_top1_balanced_accuracy:.4f}, Test Top 1 Balanced Accuracy (Country): {test_top1_balanced_accuracy_country:.4f}")
            if accuracy_per_country:
                print("Accuracy per country:")
                # Print sorted by country, sort by accuracy descending, value is key[1]
                for country, accuracy in sorted(accuracy_per_country.items(), key=lambda item: item[1], reverse=True):
                    print(f"Country {country}: {accuracy:.5f}")
        else:
            print(f"Test Loss: {test_loss:.4f}, Test Top 1 Accuracy: {test_top1_accuracy:.4f}, Test Top 3 Accuracy: {test_top3_accuracy:.4f}, Test Top 5 Accuracy: {test_top5_accuracy:.4f}")
            print(f"Test Top 1 Balanced Accuracy: {test_top1_balanced_accuracy:.4f}")
            if accuracy_per_country:
                print("Accuracy per country:")
                # Print sorted by country, sort by accuracy descending, value is key[1]
                for country, accuracy in sorted(accuracy_per_country.items(), key=lambda item: item[1], reverse=True):
                    print(f"Country {country}: {accuracy:.5f}")

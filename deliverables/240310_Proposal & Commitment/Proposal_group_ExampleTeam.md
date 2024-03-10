# Proposal for DSPRO2 (FS24) - 

Proposal for title...

1. From Pixels to Insights: Unleashing the Power of Image Classification
2. Beyond Borders: Exploring Countries and Coordinates via Images
3. Mapping the Unseen: A Journey with Image-Based Predictions

## Group Members
 [Linus Schlumberger](https://gitlab.com/Killusions)
 [Lukas Stöckli](https://gitlab.com/Valairaa)
 [Yutaro Sigrist](https://gitlab.com/yusigrist)
 
## Short Project Description
As group we never have worked a lot with Image Classification nor worked with it. So, we want to create a Image Classification Model, which is based on simple street view Images and predicts by looking at the images the country which the image represents as first step. As second step, the model will be more advanced and the Model predicts the coordinate of the image instead of predicting the right country.
Instead of addressing just one business case, the goal of this image classification model is to serve as a foundational model for multiple scenarios. But for what main purposes could an image classifier for countries or coordinates be valuable?
- **Helping find missing persons**: It can help find where missing people might be by analyzing pictures shared publicly. The emotional impact of helping reunite families or providing important clues is huge. Especially when the model will be used additionally to the finding process for a police. For missing people, every second counts after a kidnapping, especially when the person is missing internationally.
- **Supporting Humanitarian Action:** In disaster situations, it could help to quickly identify the most affected areas by analyzing current images from social media or aid organizations. This would improve the coordination of rescue and relief efforts and offer hope and support to those affected.
- **Discover New Travel Destinations:** Have you ever come across stunning images of places on Instagram or other social media platforms and wondered where they were taken? Our image classifier can help you with that. By analyzing the image, our classifier can identify the location and provide you with the information you need to plan your next visit to this amazing place. This way, you can discover new and exciting travel destinations that you may have never known about before.
## Data Description

Our dataset consists of images obtained from Street View on Google Maps. Due to the high cost of Google Maps API, we utilized an alternative approach by using the game Geoguessr, which also uses Google Street View API. This game involves guessing the country or coordinates of a given location. The game offers a cost-effective solution at only $5/Month for unlimited access. We developed a script to automatically play the game, capture the country and coordinate data, and take a screenshot of the Street View image. This data, complete with coordinates and country information, will serve as the basis dataset for our classifier.

## Cloud Service Integration

"Describe which tool you plan to use and how. For example, you may decide to do the greatest part of your training on your laptop and just run some final larger runs on the cloud, or maybe do only hyperparameter tuning in the cloud. It is ok if your final approach will be different than what you describe here. The goal of this document is to give you a more concrete starting point. Keep in mind that it is good practice to do some cost management and planning in the cloud, so you can describe how you plan to do this, too (very shortly)."


Our plan is to conduct initial training with small batch sizes locally, which will also help us test our training pipeline. However, for more intensive, large-scale training, we intend to use a Cloud Service. Currently, we are undecided on which cloud service to choose. We will conduct a comprehensive comparison of cost and performance before making a decision. All Data Quality Analysis (DQA) tasks will be performed locally on our personal computers.

## Kanban Tool
For our team, it is essential to use a Kanban tool that seamlessly integrates into our workflow and GitLab project. Our goal is to incorporate every document related to this project and store it within the GitLab repository. To achieve this, we utilize a Markdown file, which we enhance with plugins in the Obsidian application.

The primary advantage of this approach is that we can work with Markdown files in Obsidian and seamlessly synchronize them within our GitLab workflow using pull and push commands. Within these files, we organize tasks into five columns: **Backlog**, **Current**, **Doing**, **Done**, and **Archive**. The **Backlog** column contains all pending tasks, **Current** represents tasks for the immediate future, **Doing** includes tasks currently in progress, **Done** represents completed tasks (used for bug control and verification), and **Archive** is where tasks are stored after thorough review. Additionally, we can color-code tasks using hashtags, allowing us to structure our data science project workflow around key milestones that shape the project.

## Experiment Tracking Tool Approach

To keep track of our experiments and ensure reproducibility, we plan to use an experiment tracking tool. Specifically, we will use **wandb.ai**, a popular tool that allows for easy logging, visualization, and comparison of machine learning experiments. With **wandb.ai**, we can track our model's performance, hyperparameters, and other relevant metrics. This will enable us to compare different experiments and make informed decisions about which approaches to pursue further. By using **wandb.ai**, we can ensure that our experiments are well-documented and reproducible, which is essential for both research and development purposes.
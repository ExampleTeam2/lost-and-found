# Proposal for DSPRO2 (FS24) - Lost & Found: Predicting Locations from Images

## Group Members

[Linus Schlumberger](https://github.com/Killusions)
[Lukas Stöckli](https://github.com/Valairaa)
[Yutaro Sigrist](https://github.com/yusigrist)

## Short Project Description

As a group, we have never worked a lot with Image Classification. So, we want to create an Image Classification model, which is based on simple street view images grouped by countries, predicting the country an image was taken in. As a second step, we will use an additional model to predict the coordinates of the image, instead of only predicting the right country.
Instead of addressing just one business case, the goal of this Image Classification model is to serve as a foundational model for multiple scenarios. But for what main purposes could an image classifier for countries or coordinates be valuable?

- **Helping find missing persons**: Our solution can help find where missing people might be by analyzing pictures shared publicly. The emotional impact of helping reunite families or providing important clues is huge. Especially when the model will be used in addition to the search process for the police. For missing people, every second counts after a kidnapping, especially when the search is international.
- **Rediscovering memories and family history**: Have you ever come across an old image of someone close to you? Maybe of a deceased family member or someone who may just not remember where it was taken. Our model can try to predict the rough location to help you rediscover your past.
- **Supporting humanitarian action:** In disaster situations, it could help to quickly identify the most affected areas by analyzing current images from social media or aid organizations. This would improve the coordination of rescue and relief efforts and offer hope and support to those impacted.
- **Discovering new travel destinations:** Have you ever encountered stunning images of places on Instagram or other social media platforms and wondered where they were taken? Our image classifier can help you with that. By analyzing the image, our classifier can identify the location and provide you with the information you need to plan your next visit to this amazing place. This way, you can discover new and exciting travel destinations that you may have never known about before.
- **Classification as a service**: With this service, we will help other companies or data science projects label their data. Sometimes companies want to block, permit, or deploy individual versions of their applications in different countries. Some countries have more restrictions for deploying applications, therefore the image predictor can help the companies have the right version on the right devices for these countries.

## Data Description

Our dataset consists of images obtained from Street View on Google Maps. Due to the high cost of Google Maps API, as well as the challenge of distributing these images evenly while still trying to cover as much ground as possible, we didn't want to use the API directly. Many places do not have any or sparse Street View coverage, which we did not want to try to fairly represent ourselves instead, we utilized an alternative approach by using the game Geoguessr, which also uses Google Street View API. This game involves guessing the country or coordinates of a given location. The game offers a cost-effective solution at only $5/Month for unlimited access. We developed a script to automatically play the game, capture the country and coordinate data, and take a screenshot of the Street View image. This data, complete with coordinates and country information, will serve as the base dataset for our classifier.

## Cloud Service Integration

At the moment, we are utilizing a cloud service for data collection and preprocessing. We have rented a server from Hetzner for web scraping. Currently, we are scraping images from Geoguessr and have accumulated a total of 60,000 images along with their locations in just one week. As part of the preprocessing task, we plan to reduce the size of the images on the server before processing them locally on our laptops. The server costs $5 per month, excluding bandwidth. For bandwidth, we would only need to pay after using 20TB of data, so our goal is to avoid reaching this amount of data. The 60,000 images we have now amount to 90 GB of data.
Our plan for training the model and performing hyperparameter-tuning involves using a cloud service due to the large size of the entire dataset. It would be impractical to train the model locally on a laptop. We aim to conduct the more costly training on the GPU-Hub at HSLU to save costs. However, we are still undecided about which cloud service to choose. We plan to carry out a thorough comparison of cost and performance before making a decision. All Data Quality Analysis (DQA) tasks will be carried out locally on our personal computers.

## Kanban Tool

For our team, it is essential to use a Kanban tool that seamlessly integrates into our workflow and GitLab project. Our goal is to incorporate every document related to this project and store it within the GitLab repository. To achieve this, we utilize a Markdown file, which we enhance with plugins of the application Obsidian.
The main benefit of this method is that we can operate with Markdown files in Obsidian and integrate them flawlessly into our GitLab workflow for synchronized files, version control, and utilizing pull and push commands. Within these files, we organize tasks into five columns: **Backlog**, **Current**, **Doing**, **Done**, and **Archive**. The **Backlog** column contains all pending lower-priority tasks, **Current** represents tasks for the immediate future, **Doing** includes tasks currently in progress, **Done** represents completed tasks (used for bug control and verification), and **Archive** is where tasks are stored after thorough review. Additionally, we can color-code tasks using hashtags, allowing us to structure our data science project workflow around key milestones that shape the project.

## Experiment Tracking Tool Approach

To keep track of our experiments and ensure reproducibility, we plan to use an experiment tracking tool. Specifically, we will use **wandb.ai**, a popular tool that allows for easy logging, visualization, and comparison of machine learning experiments. With **wandb.ai**, we can track our model's performance, hyperparameters, and other relevant metrics. We will also store them seamlessly on our GitLab project to have version control and the statistics locally on our laptops. This will enable us to compare different experiments and make informed decisions about which approaches to pursue further. By using **wandb.ai**, we can ensure that our experiments are well-documented and reproducible, which is essential for both research and development purposes.

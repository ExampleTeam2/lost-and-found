# Proposal for DSPRO2 (FS24) - Image Classifier

## Group Members
 [Linus Schlumberger](https://gitlab.com/Killusions)
 [Lukas Stöckli](https://gitlab.com/Valairaa)
 [Yutaro Sigrist](https://gitlab.com/yusigrist)
 
## Short Project Description
As group we never have worked a lot with Image Classification nor worked with it. So, we want to create a Image Classification Model, which is based on simple street view Images and predicts by looking at the images the country which the image represents as first step. As second step, the model will be more advanced and the Model predicts the coordinate of the image instead of predicting the right country.
Instead of facing just one Business Case, the aim of this Image Classification Model is to be a base model for multiple Cases. But for what use will a classifying model for countries or coordinate from images be? 
- **Helping find missing persons**: It can help find where missing people might be by analyzing pictures shared publicly. The emotional impact of helping reunite families or providing important clues is huge. Especially when the model will be used additionally to the finding process for a police. For missing people every second counts after a kidnapping, especially when the person is missing internationally.
- **Supporting Humanitarian Action:** In disaster situations, it could help to quickly identify the most affected areas by analyzing current images from social media or aid organizations. This would improve the coordination of rescue and relief efforts and offer hope and support to those affected.
- **Find new places to visit:** We all have seen cool places on Instagram or other social platform. But often we don't know where this place is. In this case, our image classifier can help to find out where this image has been taken, and so we can, may visit the place.


But what should de Model classify? Therefore we did some research and decided us for a Classification Task which could be widely used on different important Use Cases. It should help for example find missing people by analysing openly published pictures of them after the crime.




"Briefly describe the idea behind the project."

## Data Description

"Describe briefly what data you are going to use."

Our Data is are the images from Street View (from Google Maps). Since Google Maps API is very expensive, so we use geoguessr to get to our data. This game use Street View, and you guess the country or the coordinate of this place. In comparison, this game only cost us 5USD/Month for unlimited access. So we used a script to automatic play this game, and get the country and the coordinates and take also a screenshot of the street view. 

This data with the coordinates and countries will be our data for our classifier.

## Cloud Service Integration



"Describe which tool you plan to use and how. For example, you may decide to do the greatest part of your training on your laptop and just run some final larger runs on the cloud, or maybe do only hyperparameter tuning in the cloud. It is ok if your final approach will be different than what you describe here. The goal of this document is to give you a more concrete starting point. Keep in mind that it is good practice to do some cost management and planning in the cloud, so you can describe how you plan to do this, too (very shortly)."


Our goal is to training with small batch sizes locally (also to test our training pipeline), but for the actual training (which are more intense) to use a Cloud Service. But we are not sure now, which cloud service to use. 
We will do also a cost and performance comparison before decide which cloud service to use.
All DQA will be done locally.



## Kanban Tool
For our team, it is essential to use a Kanban tool that seamlessly integrates into our workflow and GitHub project. Our goal is to incorporate every document related to this project and store it within the GitLab repository. To achieve this, we utilize a Markdown file, which we enhance with plugins in the Obsidian application.

The primary advantage of this approach is that we can work with Markdown files in Obsidian and synchronize them seamlessly in our GitLab workflow using pull and push commands. Within these files, we organize tasks using five columns: "Backlog", "Current", "Doing", "Done", and "Archive". The ‘Backlog’ column contains all pending tasks, ‘Current’ represents tasks for the immediate future, ‘Doing’ includes tasks currently in progress, and ‘Archive’ is for completed tasks. Additionally, we can color-code tasks using hashtags, allowing us to structure our data science project workflow around key milestones that shape the project.

## Experiment Tracking Tool Approach




"Describe shortly how you will use the experiment tracking tool and which one you plan to use. We use wandb.ai"
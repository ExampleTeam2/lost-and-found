# Proposal for DSPRO2 (FS24) - Image Classifier

## Group Members
 [Linus Schlumberger](https://gitlab.com/Killusions)
 [Lukas Stöckli](https://gitlab.com/Valairaa)
 [Yutaro Sigrist](https://gitlab.com/yusigrist)
 
## Short Project Description
As group we never have worked a lot with Image Classification nor worked with it. So, we want to create a Image Classification Model, which is based on simple street view Images and predicts by looking at the images the country which the image represents as first step. As second step, the model will be more advanced and the Model predicts the coordinate of the image instead of predicting the right country.
Instead of facing just one Business Case, the aim of this Image Classification Model is to be a base model for multiple Cases. But for what use will a classifying model for countries or coordinate from images be? 
- **Helping finding missing persons**: It can help find where missing people might be by analysing pictures shared publicly. The emotional impact of helping reunite families or providing important clues is huge. Especially when the model will be used additionally to the finding process for a police. For missing people every second counts after a kidnapping.
- **Supporting Humanitarian Action:** In disaster situations, it could help to quickly identify the most affected areas by analysing current images from social media or aid organisations. This would improve the coordination of rescue and relief efforts and offer hope and support to those affected.


But what should de Model classify? Therefore we did some research and decided us for a Classification Task which could be widely used on different important Use Cases. It should help for example find missing people by analysing openly published pictures of them after the crime.




"Briefly describe the idea behind the project."

## Data Description

"Describe briefly what data you are going to use."

## Cloud Service Integration



"Describe which tool you plan to use and how. For example, you may decide to do the greatest part of your training on your laptop and just run some final larger runs on the cloud, or maybe do only hyperparameter tuning in the cloud. It is ok if your final approach will be different than what you describe here. The goal of this document is to give you a more concrete starting point. Keep in mind that it is good practice to do some cost management and planning in the cloud, so you can describe how you plan to do this too (very shortly)."

## Kanban Tool
For our team, it is essential to use a Kanban tool that seamlessly integrates into our workflow and GitHub project. Our goal is to incorporate every document related to this project and store it within the GitLab repository. To achieve this, we utilize a Markdown file, which we enhance with plugins in the Obsidian application.

The primary advantage of this approach is that we can work with Markdown files in Obsidian and synchronize them seamlessly in our GitLab workflow using pull and push commands. Within these files, we organize tasks using five columns: "Backlog", "Current", "Doing", "Done", and "Archive". The ‘Backlog’ column contains all pending tasks, ‘Current’ represents tasks for the immediate future, ‘Doing’ includes tasks currently in progress, and ‘Archive’ is for completed tasks. Additionally, we can color-code tasks using hashtags, allowing us to structure our data science project workflow around key milestones that shape the project.

## Experiment Tracking Tool Approach




"Describe shortly how you will use the experiment tracking tool and which one you plan to use. We use wandb.ai"
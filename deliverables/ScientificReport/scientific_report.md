# Scientific Report

Team name: ExampleTeam

**Group Members**
 [Linus Schlumberger](https://gitlab.com/Killusions)
 [Lukas Stöckli](https://gitlab.com/Valairaa)
 [Yutaro Sigrist](https://gitlab.com/yusigrist)

```table-of-contents
title: # Table of content
style: nestedList
includeLinks: true
```

# Business Understanding
This project explores the development of an Image Classification model, focusing on simple street-view images grouped by countries to predict the country where an image was taken. Given limited prior experience with Image Classification, this initiative aims to enhance understanding and skills in this domain. The first objective is to create a model capable of identifying the country from a given image. Building upon this, a second model will be developed to predict the exact district of the image, providing a more precise location than just the country.

The main goal is to develop a robust Image Classification model that can serve as a foundational tool for various applications. This overarching objective supports the specific sub-goals of predicting the country and coordinates of an image. This leads to the question: for what main purposes could an image classifier for countries or coordinates be valuable? By exploring potential applications, the project aims to demonstrate the broader utility of the developed models in real-world scenarios.

- **Helping find missing persons**: Our solution can help find where missing people might be by analyzing pictures shared publicly. The emotional impact of helping reunite families or providing important clues is huge. Especially when the model will be used in addition to the search process for the police. For missing people, every second counts after a kidnapping, especially when the search is international.
- **Rediscovering memories and family history**: Have you ever come across an old image of someone close to you? Maybe of a deceased family member or someone who may just not remember where it was taken. Our model can try to predict the rough location to help you rediscover your past.
- **Supporting humanitarian action:** In disaster situations, it could help to quickly identify the most affected areas by analyzing current images from social media or aid organizations. This would improve the coordination of rescue and relief efforts and offer hope and support to those impacted.
- **Discovering new travel destinations:** Have you ever encountered stunning images of places on Instagram or other social media platforms and wondered where they were taken? Our image classifier can help you with that. By analyzing the image, our classifier can identify the location and provide you with the information you need to plan your next visit to this amazing place. This way, you can discover new and exciting travel destinations that you may have never known about before.
- **Classification as a service**: With this service, we will help other companies or data science projects label their data. Sometimes companies want to block, permit, or deploy individual versions of their applications in different countries. Some countries have more restrictions for deploying applications, therefore the image predictor can help the companies have the right version on the right devices for these countries.

# Literature review

## Common approaches
Follows...

### Common architectures
Follows...

#### Possible Sources
[EfficientNet Rethinking Model Scaling for CNNs](https://arxiv.org/pdf/1905.11946)
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)
[Deep Residual Learning for Image Recognition / ResNet](https://arxiv.org/pdf/1512.03385)
[Efficient ResNets: Residual Network Design](https://arxiv.org/pdf/2306.12100)
[Pigeon for calculating loss](https://arxiv.org/pdf/2307.05845)

[Data Augmentation and overfitting](https://arxiv.org/abs/2201.03299)
[Data Augmentation in Training CNNs](https://arxiv.org/pdf/2307.06855)

___
**Requirements (2-3 pages with a minimum of 7 sources):**
1. **Foundation of Knowledge:** A literature review establishes the theoretical foundation and current state of research in the field. By requiring a minimum number of sources, students are encouraged to engage deeply with existing literature, ensuring a comprehensive understanding of the subject.  
2. **Critical Thinking and Contextualization:** Analyzing and synthesizing various sources enhances critical thinking skills. It allows students to understand different perspectives and place their work within the broader context of the field.
___

# Data collection

To talk about:

Why Geoguessr?

Why Multiplayer initially

Why not just Multiplayer, mapping Singleplayer to distribution instead

Scraping flow (rate-limiting, double-joins, screenshotting, coordinate extraction, ...)

Playwright, auto-restarting & fault tolerance

Data inspection

(Error correction with more scraping)
## test 1
### test 2

#### test 3

## test 4


Follows...

___
No Requirements, it is good to show what we did to achieve our goal with data. Maybe also ask Umberto if we should include it.
___
# Data processing

## Resize of the images
We can't train the classifier using images in a high resolution, because our resources are limited, and also often images (like from missing persons) are also very low quality. So we decided to reduce the resolution, at the beginning of the processing, about the 1/4 of the original resolution of 1280p x 720p. This also helps to move the images for learning to the server or also between us and also loading takes lot less time for future processing steps. 


To talk about:

Enriching (Singleplayer coordinates, Multiplayer names)

(Issues with reverse geocoding, country name matching)

Regions Enriching (Source, Mapping)

Filtering (Color, Blur, Duplicates, Corrupt)

___
**Requirements:**
1. **Core Competency in Data Science:** Data processing is a fundamental step in any data science project. Demonstrating this process shows the student's ability to handle and prepare data for analysis, which is a critical skill in the field.
2. **Transparency and Reproducibility:** Detailing the data processing steps ensures transparency and aids in the reproducibility of the results, which are key aspects of scientific research.
___

# Methods

Follows...

To talk about:

Basic method (Cross-entropy, ...)

DIfferent pre-trained models

(Coordinates attempt)

Data augmentation

Regions with custom loss

___
**Requirements:**
1. **Understanding and Application:** This section allows students to demonstrate their understanding of various methodologies and their ability to apply appropriate techniques to their specific project.
2. **Rationale and Justification:** Discussing the methods used provides insight into the student’s decision-making process and the rationale behind choosing specific approaches.
___
# Model Validation

## Common approaches for model evaluation
Follows...

## Human baseline performance
Follows...

## Model performance on other datasets
Follows...

___
**Requirements:**
1. Ensuring Model Reliability: Model validation is crucial for assessing the accuracy and reliability of the model. This section shows how the student evaluates the performance and generalizability of their model.
2. Critical Evaluation: It encourages students to critically evaluate their model’s performance, understand its limitations, and discuss potential improvements.
___
# Machine Learning Operations (MLOps)

Follows...

To talk about:

Project structure (from last time, should run out of the box)

Issues with large amount of files

Data-Loader, Datalists and Server

Mitigation of issues on Colab (with drive, nested, files-lists and zips)

Caching of transformed data (Hashing)

Pushing to WandB (data as well, but only test and only once)

___
**Requirements:**
1. Practical Application: This section emphasizes the practical aspect of machine learning. It’s not just about building models but also about deploying them effectively in real-world scenarios.
2. Bridging Theory and Practice: It allows students to demonstrate their ability to translate theoretical knowledge into practical applications, showcasing their readiness for industry challenges.
___

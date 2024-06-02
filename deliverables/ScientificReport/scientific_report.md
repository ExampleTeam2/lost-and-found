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

# Literature review

## Business Understanding
Follows...

## Common approaches
Follows...

### Common architectures
Follows...

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

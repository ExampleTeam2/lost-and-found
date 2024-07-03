# Lost & Found: Predicting Locations from Images

## General

This is a project for HSLU (DSPRO 2).

It consists of a codebase, a report [scientific report](./deliverables/ScientificReport/scientific_report.pdf) ([source](./deliverables/ScientificReport/scientific_report.md)), and a [published dataset](https://www.kaggle.com/datasets/killusions/street-location-images/).

The project uses Markdown (and LaTeX), Python and Node.js with Typescript.

It uses Obsidian (Markdown) for [documentation](./Overview.md) and planning, it can also be viewed with any other markdown viewer (including GitHub/GitLab).

For Python dependency management it uses [poetry](https://python-poetry.org/).

To set it up do `poetry install`, to add dependencies use `poetry add`.

To run commands use `poetry shell` to spawn a subshell.

Select the `venv` after running `poetry install` for Jupyter Notebooks.

For Node.js dependency management it uses [yarn v1](https://classic.yarnpkg.com/lang/en/).

To set it up simply type `yarn`.

To see available commands, check out the `scripts` section of the `package.json` and run them using `yarn <command>`.

All project relevant commands are handled via yarn, including formatting our Python, Typescript, and Markdown files and generating our report from our Markdown source.

## Loading data from our published dataset

To load the data from our [published dataset](https://www.kaggle.com/datasets/killusions/street-location-images/):

- Ignore our server check (`DOWNLOAD_LINK=None` in `.env`)
- Put the unzipped `data(_mapped)` directory (if you want both start with mapped) into `dspro2/1_data_collection/.data`.
- Run `yarn data:import` on a unix based system (or rename them to `geoguessr_location_******.png` and `geoguessr_result_******.json`, and copy all the JSON files into `dspro2/3_data_preparation/01_enriching/.data`).
- Execute the `dspro2/3_data_preparation/99_importing/import.ipynb` notebook (make sure to set the `MAPPED` parameter correctly and only the relevant data is inside the directory).

## Collecting data

Simply run `yarn scrape:prepare`, set `GEOGUESSR_EMAIL` and `GEOGUESSR_PASSWORD` in your `.env` file, then `yarn scrape:ui` (for local testing), `yarn scrape` or `scrape:deploy` (for multiple parallel instances).

## Course Coaches

Within this module we are supervised by the following course coaches:

- [Dr. Umberto Michelucci](https://www.hslu.ch/en/lucerne-university-of-applied-sciences-and-arts/about-us/people-finder/profile/?pid=5426)
- [Dr. Ludovic Amruthalingam](https://www.hslu.ch/en/lucerne-university-of-applied-sciences-and-arts/about-us/people-finder/profile/?pid=5381)
- [Dr. Daniela Wolff](https://www.linkedin.com/in/daniela-wolff?originalSubdomain=ch)
- [Aygul Zagidullina (internal link)](https://elearning.hslu.ch/ilias/ilias.php?baseClass=ilrepositorygui&cmdNode=yo:mw:9l:17l:xt&cmdClass=ilPublicUserProfileGUI&cmd=getHTML&ref_id=6109722&back_cmd=jump2UsersGallery&user=6566820)

## Authors and acknowledgment

The whole project was done by the following students:

- [Linus Schlumberger](https://gitlab.com/Killusions)
- [Lukas Stöckli](https://gitlab.com/Valairaa)
- [Yutaro Sigrist](https://gitlab.com/yusigrist)

# Crapauduc

## Introduction

This repository is the working space for the crapauduc project. He his the result of the work of ISCD students from HEIG-VD in 2022-2023. The goal of this project is to detect the presence of a frog and triton in a crapauduc.

## Architecture

The project is hosted in different places:

- [Main Repository](https://github.com/student-GML/crapauduc)
- Colab
- Atlas

We describe the different parts of the project in the following sections.

### Main Repository

The main repository is the one you are currently reading. It contains the code and the data used for the project. The structure of the repository is described later on.

### Colab

The Google Colab is used to train the models with small data subsets located in the google drive.

### Atlas

Atlas is the name of the server used to host the data. We use it as a datalake to store the data and the models.

## Structure

The structure of the repository is the following:

- [crapauduc_previous_work/](./crapauduc_previous_work/) contains the previous report on the crapauduc project.

- [data](./data/) contains all the .csv files describing the data.

- [documentation](./documentation/) contains the documentation and helpful tutorials for the project.

- [model](./model/) contains the weight for the smallest models used in the project.

- [notebooks](./notebooks/) contains the notebooks used to train the models.

- [report](./report/) contains the report of the project.


The labels are the following :

1. triton
2. grenouille-crapaud
3. planche
4. feuille
5. souris
6. insecte

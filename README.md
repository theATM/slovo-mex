# Multi-Modal Data Fusion - Project Work: Multi-Modal Physical Exercise Classification


In this project, real multi-modal data is studied by utilizing different techniques presented during the course.
In addition, there is an optional task to try some different approaches to identify persons from the same dataset.
Open MEx dataset from UCI machine learning repository is used. Idea is to apply different techniques to recognize
physical exercises from wearable sensors and depth camera, user-independently.

## Authors


Names:
Aleksander Madajczak,
Jan Fabian

Student numbers:
2207367,
2207371

## Usage

This project is split into multiple files including notebooks and python files. 
Each Task 1-5 have a separate notebook file and a python file with functions 
and classes used in that task in the utilities directory named accordingly. 
For example task 1 is available by running notebook 'slovo_one.ipynb' and it 
has python file with all the functions called 'utilities/fun_one.py'.
Project was made in Python 3.10

## Project structure:
* README.md - project description and instructions
* requirements.txt - project python dependencies


* slovo_one.ipynb - Notebook with Task 1 completed
* slovo_two.ipynb - Notebook with Task 2 completed
* slovo_three.ipynb - Notebook with Task 3 completed
* slovo_four.ipynb - Notebook with Task 4 completed
* slovo_five.ipynb - Notebook with Task 5 completed 


* utilities - directory with python supporting files
  * fun_one.py - Functions and Classes implementing Task 1 
  * fun_two.py - Functions and Classes implementing Task 2
  * fun_three.py - Functions and Classes implementing Task 3
  * fun_four.py - Functions and Classes implementing Task 4
  * fun_five.py - Functions and Classes implementing Task 5


* .gitignore - list of files omitted by source control
* MEx - directory with the MEx dataset, please obtain it by yourself and place it there




## Description

The goal of this project is to develop user-independent pre-processing and classification models to recognize 7 different physical exercises measured by accelerometer (attached to subject's thigh) and depth camera (above the subject facing downwards recording an aerial view). All the exercises were performed subject lying down on the mat. Original dataset have also another acceleration sensor and pressure-sensitive mat, but those two modalities are ommited in this project. There are totally 30 subjects in the original dataset, and in this work subset of 10 person is utilized. Detailed description of the dataset and original data can be access in [MEx dataset @ UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/MEx#). We are providing the subset of dataset in Moodle.

The project work is divided on following phases:

1. Data preparation, exploration, and visualization
2. Feature extraction and unimodal fusion for classification
3. Feature extraction and feature-level fusion for multimodal classification
4. Decision-level fusion for multimodal classification
5. Bonus task: Multimodal biometric identification of persons

where 1-4 are compulsory (max. 10 points each), and 5 is optional to get bonus points (max. 5+5 points). In each phase, you should visualize and analyse the results and document the work and findings properly by text blocks and figures between the code. <b> Nice looking </b> and <b> informative </b> notebook representing your results and analysis will be part of the grading in addition to actual implementation.

The results are validated using confusion matrices and F1 scores. F1 macro score is given as

<br>
<br>
$
\begin{equation}
F1_{macro} = \frac{1}{N} \sum_i^N F1_i,
\end{equation}
$
<br>
<br>
where $F1_i = 2  \frac{precision_i * recall_i}{precision_i + recall_i}$, and $N$ is the number of labels.
<br>


## Learning goals

After the project work, you should

- be able to study real world multi-modal data
- be able to apply different data fusion techniques to real-world problem
- be able to evaluate the results
- be able to analyse the outcome
- be able to document your work properly

## Relevant lectures

Lectures 1-8

## Relevant exercises

Exercises 0-6

## Relevant chapters in course book

Chapter 1-14

## Additional Material

* Original dataset [MEx dataset @ UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/MEx#)
* Related scientific article [MEx: Multi-modal Exercises Dataset for Human Activity Recognition](https://arxiv.org/pdf/1908.08992.pdf)

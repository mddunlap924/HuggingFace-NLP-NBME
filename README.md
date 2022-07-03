# Hugging-Face-NLP-NBME
## Introduction
A state-of-the-art natural language processing (NLP) model is trained on a custom labeled dataset with the objective of extracting sub-text for given features. The model was developed using the Transformers from the Hugging Face API [[1](https://huggingface.co/docs/transformers/index)]. The objective of this model is to perform named-entity recognition (NER) [[2](https://en.wikipedia.org/wiki/Named-entity_recognition)] by locating character spans for predefined categories in unstructured text.  In this specific example, the data and scripts are from a subset of the files I created when competing in the National Board of Medical Examiners (NBME) - Score Clinical Patients Notes competition on Kaggle [[3](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)].

The following example shows a visualization of NER predictions for a dataset. 

show ner predictions here

show word cloud of feature text

point

## Fine-Tuning a Pre-trained Model

Fine-tuning of a pretrained model was done with a deberta-large-v3

https://www.kaggle.com/code/ruchi798/score-clinical-patient-notes-spacy-w-b

Point to where the model would be uploaded in the directory but it was not because of file size.

Link the pretrained deberta-v3-model location 






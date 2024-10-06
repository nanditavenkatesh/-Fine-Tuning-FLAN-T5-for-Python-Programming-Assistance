# Fine-tuning FLAN-T5 for Python Question-Answering

This project focuses on fine-tuning the Flan-T5 model for Python programming-related questions and answers, improving its ability to generate helpful, accurate responses. The dataset used for training was sourced from Python-tagged questions on StackOverflow. The fine-tuning was done using a high-performance environment and evaluated using both ROUGE scores and human assessments.

## Table of Contents

- [Introduction](#introduction)
- [Model](#model)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training Procedure](#training-procedure)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Usage](#usage)
- [References](#ethics)

## Introduction

With the rise of automated coding assistance tools, this project aimed to fine-tune Google's Flan-T5 model to generate accurate and contextually appropriate Python-related answers. The model was trained on a dataset of Python questions and answers from StackOverflow. The focus was to improve the model's understanding and response generation by using incremental training methods and both automated and human evaluation.

## Model

We used the **Flan-T5 base model**, a version of Google's T5 model, enhanced for 1,000+ additional tasks. Fine-tuning was performed using Hugging Face's `T5ForConditionalGeneration` model class in a PyTorch environment, with training handled by the `Trainer` API from Hugging Face.

- **Model Input**: Python programming question
- **Model Output**: Python code snippet or explanatory text

## Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) and included Python-tagged questions and answers from StackOverflow.

- **Original Size**: ~980,000 entries
- **Filtered Size**: ~320,000 entries (based on quality filtering)
- **Train/Validation/Test Split**: 
  - Training: 80%
  - Validation: 5%
  - Test: 20%

## Preprocessing

- **Cleaning**: HTML entities were converted to plain text using Python's `BeautifulSoup` library.
- **Quality Filtering**: Only answers with at least two upvotes were retained.
- **Data Diversity**: Limited to the top two answers per question.
- **Data Splitting**: Training data was divided into batches of 20,000 samples for ease of training.

## Training Procedure

Training was done in a high-performance computing environment with an L4 GPU. We employed incremental training over multiple epochs, processing 20,000 samples per batch, with 3 epochs per batch.

Key parameters:
- **Training Strategy**: Incremental training on segmented data batches
- **Batch Size**: 20,000 samples
- **Epochs per Batch**: 3 epochs
- **Token Length**: Initially set to 20 tokens, later adjusted to 256 tokens to allow for more complete answers.

## Evaluation

We used both automated and human-based evaluations to measure model performance.

- **Automated Metrics**: ROUGE scores (Rouge-1, Rouge-2, and Rouge-L) were used to assess the overlap between generated and true answers.
- **Human Evaluation**: We manually evaluated six Python questions throughout the training process to assess relevance and correctness.

Sample evaluation questions:
```python
questions = [
    "What is the difference between list and tuple?",
    "How to create a new text file in Python?",
    "How to install Numpy?",
    "How to add 2 lists together?",
    "How to remove a specific element from a list in Python?",
    "How to find the index of an element in a list in Python?"
]
```

## Results
### **Pre-fine-tuning Results**:

- The model initially performed poorly, with ROUGE-1 scores as low as 0.05.
- Example Output: "Add the first list to the second list."

### **Post-fine-tuning**:

- After fine-tuning on 20,000 samples, there was significant improvement.
- The ROUGE scores increased by approximately 125%, with an average test loss reduction to 0.83 after the first 20,000 samples.
- Example Output: More accurate explanations, but still prone to repeated phrases in some cases.

### **Final Model Performance**:

- Test Loss: 1.18
- ROUGE-1: 0.20
- ROUGE-2: 0.06
- ROUGE-L: 0.16
- Qualitative answers improved, though some repetition issues remained.

##Future Work

- **Additional Training**: Training with larger datasets or longer batches would likely improve results further.
- **Data Expansion**: Incorporating Python-related data from other sources could enhance model generalizability.
- **Human Feedback**: Integrating reinforcement learning with human feedback could refine model performance over time.

## Usage

The model and training code are provided in a Jupyter Notebook (`.ipynb`) format. To use the notebook:

1. Open the provided Jupyter notebook in **Google Colab** or any **Jupyter environment**.
2. Make sure to install all required dependencies as outlined in the notebook, including Hugging Face's `transformers` library.
3. Download the Kaggle dataset from [this link](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) and place it in the appropriate directory as indicated in the notebook.
4. Run the cells step-by-step to preprocess the data, train the model, and evaluate the outputs.

To generate answers from the fine-tuned model:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('your-finetuned-model-path')
model = T5ForConditionalGeneration.from_pretrained('your-finetuned-model-path')

input_text = "How to create a new text file in Python?"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
output = model.generate(input_ids, max_length=256)

print(tokenizer.decode(output, skip_special_tokens=True))
```

## References

- ⚡LLM 04a - Fine-tuning LLMs⚡. (n.d.). Kaggle.com. Retrieved March 28, 2024, from https://www.kaggle.com/code/aliabdin1/llm-04a-fine-tuning-llms

- Evaluate translation or summarization with ROUGE similarity score - MATLAB rougeEvaluationScore. (n.d.). Www.mathworks.com. https://www.mathworks.com/help/textanalytics/ref/rougeevaluationscore.html 

- Finetuning Large language models using QLoRA. (n.d.). Kaggle.com. Retrieved March 28, 2024, from https://www.kaggle.com/code/neerajmohan/finetuning-large-language-models-using-qlora google/flan-t5-base ·

- Hugging Face. (n.d.). Huggingface.co. https://huggingface.co/google/flan-t5-base

- Hugging Face: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated. (2023, June 19). Mark Needham. https://www.markhneedham.com/blog/2023/06/19/huggingface-max-length-generation-length-deprecated/

- LLM Evaluation Metrics: Everything You Need for LLM Evaluation - Confident AI. (n.d.). Www.confident-Ai.com. Retrieved March 28, 2024, from https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

- Python Questions from Stack Overflow. (n.d.). Www.kaggle.com.https://www.kaggle.com/datasets/stackoverflow/pythonquestions

- RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training. (n.d.). Stack Overflow. Retrieved May 9, 2024, from https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least

- Trainer. (n.d.). Huggingface.co. Retrieved May 9, 2024, from https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainer

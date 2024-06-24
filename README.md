**# AWS-GenAI-Model-Project

## LLM Model Evaluation Notebook

In this notebook, we deploy the Meta Llama 2 7B model to evaluate its text generation capabilities and domain knowledge. We use the SageMaker Python SDK for Foundation Models to deploy the model for inference.

The Llama 2 7B Foundation model performs text generation by taking a text string as input and predicting the next words in the sequence.

### Set Up

```bash
!pip install ipywidgets==7.0.0 --quiet
!pip install --upgrade sagemaker datasets --quiet
```

There are some initial steps required for setup. If you receive warnings after running these cells, you can ignore them as they won't impact the notebook's functionality. Restart the Kernel after executing these commands.

To deploy the model on Amazon SageMaker, you need to set up and authenticate the use of AWS services. Validate that your role is the SageMaker IAM role created for the project by running the next cell.

After role verification, deploy the text generation model: Meta Llama 2 7B.

```python
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id, model_version=model_version, instance_type="ml.g5.2xlarge")
predictor = model.deploy()
```

This Python code uses Amazon SageMaker's JumpStart library to deploy a machine learning model.

- Import the `JumpStartModel` class from the `sagemaker.jumpstart.model` module.
- Create an instance of the `JumpStartModel` class using the `model_id` and `model_version` variables defined in the previous cell. This object represents the machine learning model you want to deploy.
- Call the `deploy` method on the `JumpStartModel` instance. This method deploys the model on Amazon SageMaker and returns a `Predictor` object.

The `Predictor` object (`predictor`) can be used to make predictions with the deployed model. The `deploy` method will automatically choose an endpoint name, instance type, and other deployment parameters. If you see a warning about "forward compatibility, pin to model_version...", you can ignore it. Just wait for the model to deploy.

### Invoke the Endpoint, Query, and Parse Response

The model takes a text string as input and predicts the next words in the sequence. The input we send it is the prompt.

After testing the model, run the cells below to delete the model deployment.

**Important:** If you fail to run the cells below, you will run out of budget to complete the project. Verify your model endpoint was deleted by visiting the 'Endpoints' section under 'Inference' in the left navigation menu of the SageMaker dashboard.

## Model Fine-Tuning Notebook

In this notebook, we'll fine-tune the Meta Llama 2 7B large language model, deploy the fine-tuned model, and test its text generation and domain knowledge capabilities.

Fine-tuning involves taking a pre-trained language model and retraining it for a different but related task using specific data. This approach, known as transfer learning, involves transferring knowledge learned from one task to another. Large language models (LLMs) like Llama 2 7B are trained on massive amounts of unlabeled data and can be fine-tuned on domain-specific datasets, enhancing performance in that specific domain.

### Set Up

Install and import the necessary packages. Restart the kernel after executing the cell below.

```bash
!pip install --upgrade sagemaker datasets
```

### Selecting the Model to Fine-Tune

We will use the same model as in the evaluation file (Meta Llama 2 7B) to compare the outputs.

Next, we choose the training dataset text for the domain. Here, we select an IT domain expert model.

### Deploy the Fine-Tuned Model

After tuning, we deploy the model. We will compare the performance of the fine-tuned model with the pre-trained model.

### Evaluate the Pre-Trained and Fine-Tuned Models

Use the same input from the model evaluation step to evaluate the performance of the fine-tuned model and compare it with the base pre-trained model. Run the same prompts on the fine-tuned model to evaluate its domain knowledge.

Do the outputs from the fine-tuned model provide domain-specific insightful and relevant content? You can continue experimenting with the model's inputs to test its domain knowledge.

After testing the model, run the cells below to delete the model deployment.

```python
finetuned_predictor.delete_model()
finetuned_predictor.delete_endpoint()
```

**Important:** If you fail to run the cells below, you will run out of budget to complete the project.**

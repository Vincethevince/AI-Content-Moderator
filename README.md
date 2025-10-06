# AI-Content-Moderator
A scalable AI-powered content moderation tool that detects hate speech and spam in text using fine-tuned BERT models. Features a FastAPI endpoint, ethical bias analysis via SHAP, and deployment on Google Cloud. Built for real-world platforms like YouTube to ensure safe user interactions.

## Features
- **Hate Speech Detection**: Classifies text as toxic/non-toxic with >90% accuracy using BERT.
- **Ethical AI**: Bias analysis with SHAP to ensure fairness. (planned in Stage 2)
- **Scalable API**: FastAPI endpoint for real-time moderation.
- **Deployment**: Hosted on Google Cloud for scalability.

## Tech Stack (planned)
- **ML**: Hugging Face Transformers (BERT), TensorFlow, Pytorch, scikit-learn
- **Data**: Pandas, Kaggle Toxic Comment Dataset
- **API**: FastAPI
- **Deployment**: Google Cloud, Docker
- **Monitoring**: Prometheus (planned)

## Pipeline (planned)
![AI Content Moderator Pipeline](https://github.com/Vincethevince/AI-Content-Moderator/blob/main/docs/pipeline.jpg)

## Current Roadmap
- [x] Finetune BERT for baseline model -> 0.973 ROC-AUC score (evaluation used in official challenge)
- [x] Deploy baseline model in Google Cloud
- [x] Create GUI using streamlit to make the system more usable
- [ ] Test and optimize different models to improve result quality (utilize Google Collab to speed up) - ongoing
    * [x] Deploy optimized model -> 0.9854 ROC-AUC score (BERT fine-tuning with whole dataset)
    * [ ] Undersampling/Oversampling
    * [ ] Train own model without utilizing BERT
- [ ] Add user feedback regarding predictions for RLHF
- [ ] Switch datasets to include bias and introduce SHAP

## Example Usage
GUI is available here: https://ai-content-moderator-gui-822949816423.europe-west1.run.app/
If you want to test it without the GUI, you can do it by following:
```
import requests
url = "https://ai-content-moderator-822949816423.europe-west1.run.app/predict"
data = {"text": "Put your toxic comment here"}
response = requests.post(url, json=data)
print(response.json()["response"])
``` 

Or you can test it with curl:
```
curl -X POST https://ai-content-moderator-822949816423.europe-west1.run.app/predict -H "Content-Type: application/json" -d '{"text":"Put your toxic comment here"}'
```
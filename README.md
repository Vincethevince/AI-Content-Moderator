# AI-Content-Moderator
SafeText AI: A scalable AI-powered content moderation tool that detects hate speech and spam in text using fine-tuned BERT models. Features a FastAPI endpoint, ethical bias analysis via SHAP, and deployment on Google Cloud. Built for real-world platforms like YouTube to ensure safe user interactions.

## Features (planned)
- **Hate Speech Detection**: Classifies text as toxic/non-toxic with >90% accuracy using BERT.
- **Ethical AI**: Bias analysis with SHAP to ensure fairness.
- **Scalable API**: FastAPI endpoint for real-time moderation.
- **Deployment**: Hosted on Google Cloud Vertex AI for scalability.

## Tech Stack (planned)
- **ML**: Hugging Face Transformers (BERT), TensorFlow, scikit-learn
- **Data**: Pandas, Kaggle Toxic Comment Dataset
- **API**: FastAPI
- **Deployment**: Google Cloud Vertex AI or Heroku
- **Monitoring**: Prometheus (planned)

## Pipeline (planned)
![AI Content Moderator Pipeline](/docs/pipeline.png)
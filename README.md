# Welcome to our AI 3 Natural Language Processing (NLP) Github Repository!

## 💡 About

📌 Our project primarily focuses on tackling NLP-based tasks in extracting corpora surrounding mental health disorders, particularly autism. Reasons for choosing such a topic is because its traditional methods in detection is not easily noticeable, especially in its early stages. These may be prone to subjectvity, time constraints, and a resource-dependent environment. Artificial intelligence's goal is to reduce and alleviate these constraints by providing more flexibility and efficiency for clincians and mental health researchers. With that said, AI is used as a complement or auxilliary tool for the desired the mentioned professionals. One task that this project focuses is through extracting textual information related to autism including statistical data, history, potential signs and symptoms of autism, and so on as a way to reduce time in clinical diagnosis and trials for mental health researchers and clinicians. The selected model to be used as a chatbot is the Qwen2.5-1.5B-Instruct, known for its lightweight but accurate when it comes to generating information. It utilizes Low-Rank Adaptation (LoRA) for efficient fine tuning ensuring stable performance across all devices without heavy compromises. It is trained using two corpora: [A Comprehensive Book on Autism Spectrum Disorders](https://www.intechopen.com/books/463?fbclid=IwY2xjawSCzopleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeCYUI_usfkld14swXfwgfDaTbrohriNr9-MgPcLcV9CDnn2KZCOLP5krmFhk_aem_KaMi2c7OY9JjN4L43OhOWA) and [Autism Spectrum Disorder (revised)](https://www.intechopen.com/books/463?fbclid=IwY2xjawSCzopleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeCYUI_usfkld14swXfwgfDaTbrohriNr9-MgPcLcV9CDnn2KZCOLP5krmFhk_aem_KaMi2c7OY9JjN4L43OhOWA). Complementary to this, the project also utiliezd a Retrieval-Augmented Generation system or RAG to reinforce memory and reduce hallucinatory responses.

## 👥 Contributors

| Contributors                          | GitHub Username | Role/Responsibility                                                                 | LinkedIn |
|-------------------------------|-----------------|-------------------------------------------------------------------------------------|----------|
| Aaron Gabriel L. Novesteras   | [@Paradoxidus](https://github.com/Paradoxidus) | Paper writing, training & validation, testing, results analysis, presetation                     | [LinkedIn](https://www.linkedin.com/in/aaron-gabriel-novesteras-077a352a9) |
|  William Daniel D. Aguilar      | -             | Documentation (Colab notebooks, paper, supporting materials), Paper writing, training and validation, deployment, presentation                   | - |
|  Nicko Gabriel A. Baldo        | -             | Training, paper writing, deployment, presentation                                                 | - |

## 📦 Dependencies

1. **Document Processing & Vector Store**
- PyMuPDF   
- LangChain (Text Splitters)   
- FAISS (Facebook AI Similarity Search)
2. **Model & NLP Frameworks**
- Transformers
- Sentence-Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL
- NLTK
3. **Evaluation & Visualization**
- Scikit Learn   
- ROUGE   
- Matplotlib   
- Seaborn   
- Ipywidgets (For interactive chat interface)

## 📊 Results and Summary
The model was evaluated on both in-domain queries (autism-specific) and out-of-domain queries to test the safety and refusal mechanisms. The RAGAS evaluation metrics demonstrate high reliability in semantic context and factual accuracy.

| Metric            | In-Domain (Q1–Q12) | Out-of-Domain (Q13–Q24) |
|-------------------|--------------------|-------------------------|
| Precision@3       | 0.917              | 0.000                   |
| Recall@3          | 0.833              | 0.125                   |
| Answer Relevancy  | 0.845              | 0.000                   |
| Faithfulness      | 1.000              | 0.000                   |
| Context Relevancy | 0.780              | 0.505                   |
| Groundedness      | 0.825              | 0.000                   |


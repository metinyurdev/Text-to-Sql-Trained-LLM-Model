# 🚀 Text-to-SQL Model - DeepSeek-R1-Distill-Llama-8B

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Unsloth](https://img.shields.io/badge/Unsloth-Efficient%20Fine%20Tuning-blue?style=for-the-badge&logo=unsloth)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?style=for-the-badge&logo=googlecolab)

## 📌 About the Project
This project provides a fine-tuned model capable of performing **text-to-SQL** conversions using the **[unsloth/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)** model. This model generates appropriate SQL queries based on natural language questions asked by users. It is accessible via **Hugging Face Transformers** and can be used for future development of a **text-to-SQL chatbot**.

## 🚀 Features
- 🏗 **Unsloth-Based**: Optimized using the `unsloth/DeepSeek-R1-Distill-Llama-8B` model.
- 🎯 **Natural Language to SQL Generation**: Generates SQL queries based on user questions.
- 🔥 **Fine-Tuned Model**: Trained on a custom dataset.
- ☁ **Easy Integration with Transformers**: Easily integrate the model using the Hugging Face library.

## 📥 Using the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "metinyurdev/text-to-sql-trained-llm-model"

# Load Tokenizer and Model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_sql(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

query = "List total sales by country."
print(generate_sql(query))
```

## 🖥 Running on Google Colab
You can easily test the model using Google Colab.

🔗 **Colab Link:** [Colab Code Link](https://colab.research.google.com/drive/11JLPOdAddzrRmtS0ambdOIxwyZe8qflH)

## 🔗 Model on Transformers
You can review and download the model on Hugging Face.

🔗 **Model Link:** [HuggingFace Link](https://huggingface.co/metinyurdev/text-to-sql-trained-llm-model)

## 📌 Future Plans
- 🤖 Development of a **Text-to-SQL Chatbot**
- 🚀 Enhancing the model with **larger SQL datasets**
- 🛠 **Web or API integration**

## 🤝 Contributing
For any contributions and feedback, please open a **Pull Request** or create an **Issue**! 🚀

---

## 👨‍💻 Author
**Metin Yurduseven**  
🔗 [GitHub](https://github.com/metinyurdev)  
📧 metin.yurduseven@gmail.com


---
**📌 License:** MIT
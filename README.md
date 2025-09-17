# Evaluating Impact of Carbon Footprint Monitoring on Training Efficiency of Machine Learning Models with CodeCarbon and EuroSAT (RGB)

This repository contains the code, datasets, and supporting materials for the research paper:

> **Title:** Evaluating Impact of Carbon Footprint Monitoring on Training Efficiency of Machine Learning Models with CodeCarbon and EuroSAT (RGB)  
> **Authors:** Rajwant Singh Rao, Roshni Sonwani, Saket Sinha, and Alok Mishra  
> **Corresponding Author:** Dr. Alok Mishra (NTNU, Norway)  
> **Affiliation:**  
> - Department of Computer Science and Information Technology, Guru Ghasidas Vishwavidyalaya, Bilaspur, India  
> - Faculty of Engineering, NTNU-Norwegian University of Science and Technology, Norway

This study explores **Green AI practices** by evaluating the carbon footprint and energy consumption of lightweight deep learning models trained on the **EuroSAT RGB satellite image dataset**.  
The key innovation is **real-time CO₂ and energy tracking** using the [CodeCarbon](https://github.com/mlco2/codecarbon) tool.

---

## 📘 Abstract
The growing computational demands of machine learning have raised concerns about the environmental sustainability of artificial intelligence (AI), especially in domains like remote sensing.  
This project benchmarks four lightweight deep learning architectures:
- **MobileNetV3**
- **EfficientNet-Lite0**
- **H-NAS (Simulated Hypernetwork NAS)**
- **Custom Lightweight CNN**

Real-time tracking of emissions and energy usage is conducted using CodeCarbon.  
The **EuroSAT RGB dataset** is used for land-use classification, demonstrating that high accuracy can be achieved with **minimal carbon emissions** and **low energy consumption**.

---

## 🌍 Dataset

### EuroSAT RGB Dataset
- **Source:** Sentinel-2 satellite imagery  
- **Classes:** 10 land-use classes (e.g., Forest, Residential, Industrial, Rivers, etc.)  
- **Number of Images:** 27,000  
- **Image Resolution:** 64×64 pixels  
- **License:** Open Access  

🔗 **Dataset Link:** [https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)

---

## ⚙️ Features
- Real-time tracking of **energy consumption** and **CO₂ emissions** during model training using CodeCarbon.
- Comparative analysis of four lightweight models for **Green AI**.
- Introduction of a **Green Score metric**: Accuracy per gram of CO₂ emitted.
- Dataset reduction and energy optimization using:
  - **Principal Component Analysis (PCA)**
  - **Clustering**
  - **Quantization**
  - **Pruning**
- Environmentally conscious evaluation and reporting framework for sustainable AI.

---

## 🛠️ Software and Tools

| Software / Tool | Purpose |
|-----------------|---------|
| [Python 3.8+](https://www.python.org/) | Programming language |
| [TensorFlow](https://www.tensorflow.org/) | Model development & training |
| [PyTorch](https://pytorch.org/) | Alternative model training framework |
| [CodeCarbon](https://github.com/mlco2/codecarbon) | Real-time CO₂ and energy tracking |
| [Scikit-learn](https://scikit-learn.org/) | Data preprocessing (PCA, clustering) |
| [Matplotlib](https://matplotlib.org/) | Visualization |

---

## 🧩 Project Structure

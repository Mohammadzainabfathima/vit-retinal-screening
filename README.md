# Vision Transformer-Based Retinal Disease Screening on RFMiD

Individual project for MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.
Author: Zainab Fathima Mohammed (ID 1710746).

A ViT-B/16 is fine-tuned to screen retinal fundus photographs as healthy or abnormal. The project also
analyses performance across subgroups (camera, brightness, contrast, disease) and discusses ethical
implications. This is a research prototype, not a medical device.

## Results (test set, 640 photographs)

| Metric | Value |
| --- | --- |
| AUC | 0.970 |
| Sensitivity | 0.915 (463 / 506) |
| Specificity | 0.918 (123 / 134) |

Sensitivity differs by camera proxy (image size): 0.24 on 2048x1536 (n = 73), 0.86 on 4288x2848
(n = 128) and 0.96 on 2144x1424 (n = 439). See the report for the analysis and its limits.

## Files

| File or folder | Purpose |
| --- | --- |
| `day3_rfmid_train_full-3.ipynb` | Data preparation, hyperparameter search, training, evaluation |
| `day4_rfmid_bias_attention-4.ipynb` | Subgroup bias analysis, threshold analysis, attention heatmaps |
| `app.py`, `app_config.json` | Streamlit demo (run with `streamlit run app.py`) |
| `day4/` | Bias, threshold and per-disease CSV tables and figures, attention heatmaps |
| `drive-download-20261022T155306Z-1-001/` | Training, evaluation and hyperparameter-search results (CSV, JSON, PNG) |
| `requirements.txt` | Python packages |

Images and trained weights (`vit_best.pt`) are not stored here because of their size.

## How to run

1. Open the two notebooks in Google Colab (GPU runtime) and run them in order. The dataset is downloaded
   from Hugging Face (`ctmedtech/RFMID`).
2. The training notebook saves `vit_best.pt` and `app_config.json`. Put them next to `app.py`.
3. Install packages and start the demo:

```
pip install -r requirements.txt
streamlit run app.py
```

## Data

RFMiD: Pachade, S., et al. (2021). Retinal Fundus Multi-Disease Image Dataset (RFMiD): A dataset for
multi-disease detection research. Data, 6(2), 14. Released under CC BY 4.0.


## Disclaimer

This model is for study purposes only. It must not be used for medical decisions.

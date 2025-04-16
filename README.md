# EEG-Emotion-Models-GUI implementation branch
----------------------------------------------------------------------

# 🧠 EEG Emotion Recognition Models

_A Research Tool for Brainwave-Based Emotion Analysis_

Emotion recognition DL/ML models for EEG and multimodal variations. Currently only for DREAMER dataset. Uses LOGO.
This open-source project blends computational neuroscience and machine learning to identify human emotional states from EEG data. Built with a focus on transparency, reproducibility, and modular experimentation.

---

## ✨ Core Capabilities

- 🧠 **Frequency Band Analysis** — Goes trough all 5 different frequency bands and the overall signal.
- 🧬 **Many Models** — SVM, RF, CNN, Fuzzy-CNN, Attention-Enhanced Fuzzy-CNN, Fuzzy Domain-Adversarial Network and GraphCNN.
- 📉 **Multimodal Feature Extraction** — Includes other modalities, for now only ECG.
- 🧪 **Offline** — Analyze recorded EEG or stream from devices.

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/YKesX/EEG-Emotion-Models.git
cd EEG-Emotion-Models

# Install dependencies
pip install -r requirements.txt

# For CUDA-enabled installations(optional)
pip install numpy pandas scipy scikit-learn matplotlib
pip install tensorflow-gpu>=2.5.0
pip install spektral
pip install torch>=1.8.0 torchvision>=0.9.0 torchaudio>=0.8.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Run main pipeline(depending on your preference)
python AIOtensorflow.py / python AIOpytorch.py
```

## ✅ To Do

✅ Adding GUI
✅ Adding pausing ability
✅ Adding in selection for which model to train

- [ ] Add more modalities(need more datasets)
- [ ] Real-Time EEG analysis
- [ ] Include additional datasets in `data/`
- [ ] Integrate more model variations

---

## 📜 License

This project is released under a **dual-license model**:

- 🧩 **[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)** — Ideal for academic and personal use. Must disclose source for derivatives.
- 🏢 **Commercial License** — Proprietary usage, patent rights, integration with non-GPL software? [Contact us](mailto:yagizhankeskin91@protonmail.com).

> 📌 Note: GPLv3 usage **does not** grant access to any related patents unless explicitly licensed.

---

## 📚 Citations & Academic Use

If this work contributes to your research, please cite it:

```bibtex
@misc{eegemotion2025,
  author       = {Yağızhan Keskin},
  title        = {EEG Emotion Recognition Models},
  year         = {2025},
  howpublished = {\url{https://github.com/YKesX/EEG-Emotion-Models}},
  note         = {Version 1.0}
}
```

---

## 🤝 Contributing Guidelines

We welcome pull requests, feature ideas, and feedback. To contribute:

1. Fork this repo
2. Create a new branch
3. Make your edits
4. Submit a PR with context

> 🧠 Contributors agree their changes may be dual-licensed under GPLv3 and a commercial license.

---

## ⚠️ Disclaimer

This project is a **non-medical research tool** provided **"as is"** without any warranties. It is not approved for clinical or safety-critical applications. Use with informed discretion.
```

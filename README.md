# Unsupervised Domain-Adversarial Transfer Learning (For Educational Use)

> This repository and its example codes are intended **for educational and research learning purposes**, demonstrating common **Unsupervised Domain Adaptation (UDA)** methods and their simplified implementations. If this work helps you, please consider giving it a Star 🌟.

---

## Objective and Purpose

This project demonstrates how to transfer labeled information from a **source domain** to an **unlabeled target domain** to improve target performance. It covers five representative UDA methods, with brief theoretical summaries and PyTorch-style reference code (educational skeletons suitable for small experiments):

* Deep Adaptation Networks (DAN) — Multi-Kernel Maximum Mean Discrepancy (MK-MMD) alignment.
* Domain-Adversarial Neural Networks (DANN) — Domain adversarial training via Gradient Reversal Layer (GRL).
* Maximum Classifier Discrepancy (MCD) — Two classifiers maximize/minimize discrepancy on target samples.
* Joint Adaptation Networks (JAN) — Align joint distributions across multiple layers using Joint MMD (JMMD).
* Cluster Alignment with a Teacher (CAT) — Teacher provides pseudo-labels and cluster centers for student alignment.
  

---

## References (Method Sources)

* Long M, Cao Y, Wang J, et al. *Learning transferable features with deep adaptation networks*. ICML 2015. (DAN / MK-MMD)
* Ganin Y, Ustinova E, Ajakan H, et al. *Domain-adversarial training of neural networks*. JMLR 2016. (DANN / GRL)
* Saito K, Watanabe K, Ushiku Y, et al. *Maximum classifier discrepancy for unsupervised domain adaptation*. CVPR 2018. (MCD)
* Long M, Zhu H, Wang J, et al. *Deep transfer learning with joint adaptation networks*. ICML 2017. (JAN / JMMD)
* Deng Z, Luo Y, Zhu J. *Cluster alignment with a teacher for unsupervised domain adaptation*. ICCV 2019. (CAT)
* Domain adversarial graph convolutional network for fault diagnosis under variable working conditions[J]. TIM 2021
* Adversarial Entropy Optimization for Unsupervised Domain Adaptation

---

## Citation

If this work or method is helpful for your research, please cite the following papers:

1. **Wei Kang, Maoxuan Zhou, Yu Guo, Tianfu Li, Jiandong Li, Yuwei Liu**,  
   *Generative adversarial augmented multi-scale CNN for machine fault diagnosis*,  
   **Control Engineering Practice**, Volume 167, 2026, 106625.  
   DOI: 10.1016/j.conengprac.2025.106625

2. **Kun He, Wei Kang, Yu Guo, Maoxuan Zhou, Jiawei Fan, Xin Chen**,  
   *Second-Order Cyclostationary Signal Extraction Method With Combined MOMEDA and Multiband Information Enhancement*,  
   **IEEE Transactions on Instrumentation and Measurement**, vol. 74, pp. 1–12, 2025.  
   DOI: 10.1109/TIM.2025.3541794

---

### BibTeX citations

```bibtex
@article{Kang2026GAAMSCNN,
  title={Generative adversarial augmented multi-scale CNN for machine fault diagnosis},
  author={Kang, Wei and Zhou, Maoxuan and Guo, Yu and Li, Tianfu and Li, Jiandong and Liu, Yuwei},
  journal={Control Engineering Practice},
  volume={167},
  pages={106625},
  year={2026},
  issn={0967-0661},
  doi={10.1016/j.conengprac.2025.106625}
}

@ARTICLE{10904169,
  author={He, Kun and Kang, Wei and Guo, Yu and Zhou, Maoxuan and Fan, Jiawei and Chen, Xin},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Second-Order Cyclostationary Signal Extraction Method With Combined MOMEDA and Multiband Information Enhancement}, 
  year={2025},
  volume={74},
  pages={1-12},
  keywords={Feature extraction;Deconvolution;Kurtosis;Vibrations;Noise;Fault diagnosis;Cyclostationary process;Interference;Filtering algorithms;Entropy;Fault diagnosis;multiband information;multipoint optimal minimum entropy deconvolution (MED) adjusted;second-order cyclostationary;shock signal enhancement},
  doi={10.1109/TIM.2025.3541794}
}
```

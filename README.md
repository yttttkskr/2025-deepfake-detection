# 2025-deepfake-detection

该项目主要完成
1. 深度伪造人脸领域的数据集对比及总结（附含数据集论文及地址）
2. 汇总2025年CVPR, ICCV, AAAI, NeurIPS, ICASSP上有关deepfake的论文

## 数据集

### 经典数据集

| 数据集 | 年份 | 真实数量 | 伪造数量 | 伪造方法数量 | 
| :----: | :----: | :----: | :----: | :----: |
| UADFV | 2018 | 49 | 49 | 1   |
| Deepfake-Timit | 2018 | 320 | 620 | 1  |
| FaceForensics++ | 2019 | 1000 | 4000 | 4  |
| DFFD | 2019 | 1000 | 3000 | 多种方法  |
| Celeb-DF v2 | 2020 | 590 | 5639 | 1  |
| DFDC | 2020     | |
| DeeperForensic-1.0 | 2020 | 50,000 | 10,000 | 1, 加 7 种扰动形式  |
| FakeAVCeleb | 2020 | 500 | 19,500 | 5  |
| WildDeepfake | 2020 | 0 | 707 | 未具体指出  |
| ForgeryNet | 2021 | 99,630 | 121,617 | 15  |
| KoDF | 2021 | 62,166 | 175,776 | 6  |
| LAV-DF | 2022 | 540 | 6,480 | 9  |
| AV-Deepfake1M | 2023 | 286,721 | 860,039 | 1  |
| DGM4 | 2023 | 230,000(图片) | 185,267(图片) | 7 |
| FFHQ-UV | 2023 | 0 | 54,165 | 3  |
| DeepFakeFace (DFF) | 2023 | 30,000(图片) | 90,000(图片) | 3  |
| AI-face | 2025 | 400,000(图片) | 1,200,000(图片) | 37  |

### 新一代多模态数据集

| 数据集 | 年份 | 数据类型及规模 | 特点 | 
| :----: | :----: | :----: | :----: |
| DD-VQA | 2024 | 2,968 图像、14,782 问答对 | 开创了VQA(Visual Question Answering)范式在深度伪造检测中的应用,将任务从分类提升到理解 |
| LOKI | 2024 | 18K+视频、图像、3D、文本和音频 | 构建了大规模基准,主要应用于合成数据检测领域,用于评测和开发多模态大模型在该任务上的能力,推动通用AI解决该问题 |
| ExDDV | 2025 | 5.4K视频、文本描述和点击标记 | 提出了一个系统化的可解释性框架,通过层次化属性使模型的解释更规范、更细粒度、更易于验证 |
| DDL | 2025 | 1.4M+视频、多层次标注 | 提升检测模型的可解释性和精准定位能力,已用于IJCAI 2025挑战赛 |

###  论文及数据集地址

https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform


https://www.idiap.ch/en/scientific-research/data/deepfaketimit


https://github.com/ondyari/FaceForensics/tree/master/dataset


https://cvlab.cse.msu.edu/project-ffd.html


https://github.com/yuezunli/celeb-deepfakeforensics


https://www.kaggle.com/c/deepfake-detection-challenge/data 


https://github.com/EndlessSora/DeeperForensics-1.0


https://github.com/DASH-Lab/FakeAVCeleb


https://github.com/OpenTAI/wild-deepfake


https://yinanhe.github.io/projects/forgerynet.html


https://deepbrainai-research.github.io/kodf/


https://github.com/ControlNet/LAV-DF


https://github.com/ControlNet/AV-Deepfake1M


https://github.com/rshaojimmy/MultiModal-DeepFake 


https://github.com/csbhr/FFHQ-UV


https://github.com/OpenRL-Lab/DeepFakeFace


https://github.com/Purdue-M2/AI-Face-FairnessBench


https://github.com/Reality-Defender/Research-DD-VQA


https://github.com/opendatalab/LOKI?tab=readme-ov-file


https://github.com/vladhondru25/ExDDV


https://deepfake-workshop-ijcai2025github.io/main/index.html

## 2025部分顶会顶刊相关模型

#### CVPR(main conference)

- Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning(https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.pdf)
- Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model(https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.pdf)
- FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing(https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf)
- SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model(https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_SIDA_Social_Media_Image_Deepfake_Detection_Localization_and_Explanation_with_CVPR_2025_paper.pdf)
- Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted(https://openaccess.thecvf.com/content/CVPR2025/papers/Yuan_Where_the_Devil_Hides_Deepfake_Detectors_Can_No_Longer_Be_CVPR_2025_paper.pdf)
- D^3: Scaling Up Deepfake Detection by Learning from Discrepancy(https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf)
- Circumventing shortcuts in audio-visual deepfake detection datasets with
unsupervised learning(https://openaccess.thecvf.com/content/CVPR2025/papers/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.pdf)


#### ICCV(main conference)

- Bi-Level Optimization for Self-Supervised AI-Generated Face Detection(https://openaccess.thecvf.com/content/ICCV2025/papers/Zou_Bi-Level_Optimization_for_Self-Supervised_AI-Generated_Face_Detection_ICCV_2025_paper.pdf)
- FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos(https://openaccess.thecvf.com/content/ICCV2025/papers/Li_FakeRadar_Probing_Forgery_Outliers_to_Detect_Unknown_Deepfake_Videos_ICCV_2025_paper.pdf)
- Intra-modal and Cross-modal Synchronization for Audio-visual Deepfake Detection and Temporal Localization(https://openaccess.thecvf.com/content/ICCV2025/papers/Anshul_Intra-modal_and_Cross-modal_Synchronization_for_Audio-visual_Deepfake_Detection_and_Temporal_ICCV_2025_paper.pdf)
- DeepShield: Fortifying deepfake video detection with local and global forgery analysis(https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_DeepShield_Fortifying_Deepfake_Video_Detection_with_Local_and_Global_Forgery_ICCV_2025_paper.pdf)
- Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection(https://openaccess.thecvf.com/content/ICCV2025/papers/Nguyen_Vulnerability-Aware_Spatio-Temporal_Learning_for_Generalizable_Deepfake_Video_Detection_ICCV_2025_paper.pdf)
- NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping(https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_NullSwap_Proactive_Identity_Cloaking_Against_Deepfake_Face_Swapping_ICCV_2025_paper.pdf)
- Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection(https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Generalization-Preserved_Learning_Closing_the_Backdoor_to_Catastrophic_Forgetting_in_Continual_ICCV_2025_paper.pdf)
- Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection(https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Beyond_Spatial_Frequency_Pixel-wise_Temporal_Frequency-based_Deepfake_Video_Detection_ICCV_2025_paper.pdf)
- Open-Unfairness Adversarial Mitigation for Generalized Deepfake Detection(https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Open-Unfairness_Adversarial_Mitigation_for_Generalized_Deepfake_Detection_ICCV_2025_paper.pdf)
- FaceShield: Defending Facial Image against Deepfake Threats(https://openaccess.thecvf.com/content/ICCV2025/papers/Jeong_FaceShield_Defending_Facial_Image_against_Deepfake_Threats_ICCV_2025_paper.pdf)


#### AAAI

- Critical Forgetting-Based Multi-Scale Disentanglement for Deepfake Detection
- Multi-modal Deepfake Detection via Multi-task Audio-Visual Prompt Learning
- ODDN: Addressing Unpaired Data Challenges in Open-World Deepfake Detection on Online Social Networks
- Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes
- Multi-View Collaborative Learning Network for Speech Deepfake Detection
- Exploring Unbiased Deepfake Detection via Token-Level Shuffling and Mixing
- Do Not DeepFake Me: Privacy-Preserving Neural 3D Head Reconstruction Without Sensitive Images. 
- Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection
- C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection
- Region-Based Optimization in Continual Learning for Audio Deepfake Detection
- GODDS: The Global Online Deepfake Detection System

#### NeurIPS

- From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting Face Deepfakes
- $X^2$-DFD: A framework for e$X$plainable and e$X$tendable Deepfake Detection
- Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes
- Fair Deepfake Detectors Can Generalize
- Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation
- MLEP: Multi-granularity Local Entropy Patterns for Generalized AI-generated Image Detection
- Unmasking Puppeteers: Leveraging Biometric Leakage to Expose Impersonation in AI-Based Videoconferencing
- The Future Unmarked: Watermark Removal in AI-Generated Images via Next-Frame Prediction
- DiffBreak: Is Diffusion-Based Purification Robust?
- VLForgery Face Triad: Detection, Localization and Attribution via Multimodal Large Language Models
- Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection
- Towards Reliable Identification of Diffusion-based Image Manipulations

#### ICASSP

- Wave-Spectrogram Cross-Modal Aggregation for Audio Deepfake Detection(https://ieeexplore.ieee.org/document/10890563)
- Integrating Spectro-Temporal Cross Aggregation and Multi-Scale Dynamic Learning for Audio Deepfake Detection(https://ieeexplore.ieee.org/document/10889337)
- Investigating voiced and unvoiced regions of speech for audio deepfake detection(https://ieeexplore.ieee.org/document/10890861)
- Fooling The Forgers: A Multi-Stage Framework for Audio Deepfake Detection(https://ieeexplore.ieee.org/document/10888175)
- Deepfake Detection of Singing Voices With Whisper Encodings(https://ieeexplore.ieee.org/document/10887871)
- Continual Unsupervised Domain Adaptation for Audio Deepfake Detection(https://ieeexplore.ieee.org/document/10890538)
- What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain(https://ieeexplore.ieee.org/document/10887568)
- Robust Audio Deepfake Detection using Ensemble Confidence Calibration(https://ieeexplore.ieee.org/document/10889972)
- RAW Data: A Key Component for Effective Deepfake Detection(https://ieeexplore.ieee.org/document/10887800)
- Reduced Spatial Dependency for More General Video-level Deepfake Detection(https://ieeexplore.ieee.org/document/10888190)
- Audio Features Investigation for Singing Voice Deepfake Detection(https://ieeexplore.ieee.org/document/10888452)
- Robust Deepfake Detection via Perturbation Domain Alignment(https://ieeexplore.ieee.org/document/10890870)
- Partial Reconstruction Error for Deepfake Detection(https://ieeexplore.ieee.org/document/10889075)
- MIFAE-Forensics: Masked Image-Frequency AutoEncoder for DeepFake Detection(https://ieeexplore.ieee.org/document/10889125)
- Freeze and Learn: Continual Learning with Selective Freezing for Speech Deepfake Detection(https://ieeexplore.ieee.org/document/10889357)
- Generalize Audio Deepfake Algorithm Recognition via Attribution Enhancement(https://ieeexplore.ieee.org/document/10889399)
- Audio-Visual Deepfake Detection With Local Temporal Inconsistencies(https://ieeexplore.ieee.org/document/10889087)
- Forensics Analysis of Residual Noise Texture in digital Images for Detection of Deepfake(https://ieeexplore.ieee.org/document/10887712)
- SpecViT: A Custom Vision-Transformer based Approach for Audio Deepfake Detection(https://ieeexplore.ieee.org/document/10889022)
- PET: High-Frequency Temporal Self-Consistency Learning for Partially Deepfake Audio Localization(https://ieeexplore.ieee.org/document/10889913)
- Towards Interactive Deepfake Analysis(https://ieeexplore.ieee.org/document/10888337)
- Unveiling Deepfakes with Latent Diffusion Counterfactual Explanations(https://ieeexplore.ieee.org/document/10890239)
- Identity-Agnostic Learning for Deepfake Face Detection(https://ieeexplore.ieee.org/document/10888411)
- From Voices to Beats: Enhancing Music Deepfake Detection by Identifying Forgeries in Background(https://ieeexplore.ieee.org/document/10890293)
- Easy, Interpretable, Effective: openSMILE for voice deepfake detection(https://ieeexplore.ieee.org/document/10890543)
- Adversarial Training and Gradient Optimization for Partially Deepfake Audio Localization(https://ieeexplore.ieee.org/document/10890470)
- Knowledge-Guided Prompt Learning for Deepfake Facial Image Detection(https://ieeexplore.ieee.org/document/10889149)
- IdentityLock: An Identity-aware Backdoor strategy for Face Swapping Defense(https://ieeexplore.ieee.org/document/10890392)
- Generalizable Audio Deepfake Detection via Latent Space Refinement and Augmentation(https://ieeexplore.ieee.org/document/10888328)
- Leveraging Mixture of Experts for Improved Speech Deepfake Detection(https://ieeexplore.ieee.org/document/10890398)
  

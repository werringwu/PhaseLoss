# Rethinking Frequency-Phase Mismatch as Structural Displacement: Phase Consistency Constraint for Low-Light Restoration (PCL)
The code will be released soon.

> **Abstract:** *Low-light image restoration (LLIR) aims to recover structural and visual fidelity from underexposed images, which is essential for robust perception in downstream vision tasks. Existing methods primarily enhance amplitude (e.g., brightness), but often overlook phase information, which plays a critical role in preserving structural consistency. This paper rethinks the low-light restoration problem by addressing the frequency-phase mismatch, modeling phase misalignment as structural displacement. Through theoretical derivations and controlled visual experiments, we demonstrate that inaccurate phase reconstruction, even with correct amplitude, results in degraded structural alignment. 
To address this, we propose a novel Phase Consistency Constraint (PCC), formulated as Phase Congruence Loss (PCL), which enforces structure-aware phase alignment in the Fourier domain.
This loss function utilizes remainder-based reparameterization and log-scaled gradients for stable optimization. Our approach is plug-and-play, architecture-agnostic, and inference-free. Experiments on multiple benchmarks demonstrate that enforcing phase consistency improves both visual quality and downstream task performance, highlighting the effectiveness of our consistency-based formulation for LLIR.*

## Background


## Main FlowChart

<details close>
<summary><b>Ablation</b></summary>


</details>


<details close>
<summary><b>Ablation</b></summary>


</details>




## Acknowledgement
This project is based on [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer), [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [HWMNet](https://github.com/FanChiMao/HWMNet), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [WaveMamba](https://github.com/AlexZou14/Wave-Mamba) and [FourLLIE](https://github.com/wangchx67/FourLLIE).

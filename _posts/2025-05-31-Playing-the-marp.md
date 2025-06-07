---
title : "Playing the Marp"
date: 2025-05-31
categories: [Marp]
tags: [Marp, Presentation, Markdown]
---

# 🎤 Creating Slide Decks with Marp

Marp (Markdown Presentation Ecosystem) allows you to write slide decks in plain Markdown and turn them into beautiful presentations. It’s simple, flexible, and integrates seamlessly with your development workflow.

---

## 🚀 Installation Guide

To get started with Marp CLI on a Debian-based system:

```bash
sudo apt update
sudo apt install nodejs npm
sudo npm install -g @marp-team/marp-cli

```


## 👀 Preview & Export
```
1. marp --preview presentation.md # for preview
2. marp --pdf presentation.md # as PDF
3. marp --pptx presentation.md # as ppt
```


🎨 Using Beamer Theme

```
1. git clone https://github.com/rnd195/my-marp-themes.gi
2. marp test.md --theme-set ./my-marp-themes -o test.pdf
```

## Tips and tricks
- Use horizontal rules (---) to separate slides.
- Use <!-- _class: lead --> or <!-- _class: invert --> to style slides.
- You can customize themes or use built-in ones like default, gaia, or uncover.
- A detailed blog [here](https://dev.to/andyhaskell/write-your-tech-talk-slides-rapidly-with-marp-2c7g)


### Example 

Below is a simple example demonstrating the use of different text sizes, image embedding, and aligning text with images in slides. You can view the output in both [HTML format](/assets/html/marp/ai_software_for_gpu.html) and [PDF format](/assets/pdf/marp/ai_software_for_gpu.pdf). To compile the Markdown into a PDF using Marp with a custom theme, run:  

```bash
marp --allow-local-files ai_software_for_gpu.md --theme-set ./my-marp-themes -o ai_software_for_gpu.pdf

```

Marp content for AI Software and GPU Evolution slides

```Markdown
--                                                                                 
  marp: true                                                                          
  theme: beam                                                                         
  paginate: true                                                                      
  size: 4:3                                                                           
  header: AI Software & GPU Industry                                                  
  footer: June 2025 · Dinesh                                                          
  title: AI Software and the GPU Industry                                             
  ---                                                                                 

  # AI Software and the GPU Industry                                                  
  ## A Symbiotic Evolution                                                            

  **June 2025 · Dinesh**                                                              

  ---                                                                                 

  ## The GPU-AI Foundation                                                            

  **Why GPUs Power AI:**                                                              
  - Thousands of CUDA cores vs. CPU's ~10-64 cores                                    
  - Massive parallelism for tensor operations                                         
  - Optimized for matrix multiplies and convolutions                                  
  - High memory bandwidth (HBM3, GDDR6X)                                              

  **Software Dependencies:**                                                          
  - Frameworks rely on GPU libraries (cuDNN, cuBLAS)                                  
  - Tensor Cores accelerate mixed-precision compute                                   
  - SIMT execution model matches AI workloads                                         

  ---                                                                                 

  ## Software Drives Hardware Innovation                                              

  <style scoped>                                                                      
  section p, section ul, section table {                                              
    font-size: 22px;                                                                  
  }                                                                                   
  </style>                                                                            

  **AI Software Explosion:**                                                          
  - Large models: GPT, Gemini, Claude demand massive compute                          
  - Frameworks push GPU limits: DeepSpeed, vLLM, Triton                               
  - Edge AI opens new markets: Jetson, CoreML                                         

  **Software Stack Impact:**                                                          
  | Layer | Examples | GPU Impact |                                                   
  |-------|----------|------------|                                                   
  | Frameworks | PyTorch, JAX | Dynamic compute graphs |                              
  | Compilers | XLA, Triton | Kernel fusion optimization |                            
  | Inference | TensorRT, vLLM | Latency-optimized compute |                          
  | Infrastructure | Ray, KServe | Multi-GPU scalability |                            

  ---                                                                                 

  ## Hardware-Software Co-Evolution                                                   

  <style scoped>                                                                      
  .content-with-image {                                                               
    display: flex;                                                                    
    align-items: flex-start;                                                          
    gap: 2rem;                                                                        
    margin-top: 1rem;                                                                 
  }                                                                                   
  .text-content {                                                                     
    flex: 1;                                                                          
  }                                                                                   
  .image-content {                                                                    
    flex: 0 0 40%;                                                                    
  }                                                                                   
  .image-content img {                                                                
    width: 100%;                                                                      
    height: auto;                                                                     
  }                                                                                   
  </style>                                                                            

  <div class="content-with-image">                                                    
  <div class="text-content">                                                          

  **Feedback Loop:**                                                                  
  
  AI Software Needs → GPU Architecture Changes                                        
       ↑                        ↓                                                     
  Performance Bottlenecks ← New Hardware Features                                     

  **Real Examples:**                                                                  
  - **Mixed-precision:** FP16, bfloat16, FP8 support                                  
  - **Communication:** NCCL, NVLink for multi-GPU                                     
  - **Memory:** SRAM improvements for transformer models                              
  - **Specialized units:** Transformer Engines in Hopper                              

  </div>                                                                              
  <div class="image-content">                                                         
  ![Co-Evolution Cycle](image.png)                                                    
  </div>                                                                              
  </div>                                                                              
  ---                                                                                 
  
  ## Future Trends & Summary                                                          

  **Emerging Trends:**                                                                
  - Open-source GPU stacks (ROCm, Triton)                                             
  - Multi-backend compilers (IREE, TVM)                                               
  - Cross-hardware abstractions                                                       
  - Energy-efficient "Green AI"                                                       

  **Key Takeaways:**                                                                  
  - AI software and GPUs are deeply interdependent                                    
  - Software innovation drives GPU adoption and design                                
  - GPUs enable software breakthroughs through scale                                  
  - Co-evolution defines the performance frontier                                     

  ---                                                                                 

  # Thank You                                                                         

  **Dinesh**                                                                          
  📧 dineshkumarb@gmail.com                                                           
  🌐 https://dkbhaskaran.github.io/                                                   
  📞 +1 999 999 9999                                         
```

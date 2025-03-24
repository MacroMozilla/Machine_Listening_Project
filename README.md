# Machine_Listening_Project

# CS-GY 9223 G: Machine Listening — Final Project Guide

**Also listed as**:  
- ECE-GY 9173 I  
- CUSP-GX 9113 C  

**Instructor**: Juan Pablo Bello  
**TA**: Julia Wilkins  

---

## 📌 Project Overview

The final project contributes **30%** of your total course grade and is divided as follows:

| Component            | Due Date   | Weight |
|---------------------|------------|--------|
| Project Proposal     | April 1    | 10%    |
| Completed Project    | May 6      | 20%    |

---

## ✅ What To Do Now

- **Form a Team**:  
  - Projects are to be done in **teams of 3 students**.  
  - Self-organize into groups via the Ed platform.

- **Choose a DCASE Task**:  
  - Select a task from the **2022–2024** editions of the **DCASE Challenge**  
  - Ensure you clearly specify the **year** and **task number** in your presentation  
  - Tasks vary by year, so pick one that interests your team

---

## 📅 Due April 1 — Project Proposal

### 📖 Presentation Content

Your slide deck (max **15 slides**, with up to **5 supplementary**) should cover:

1. **Task Definition**
   - What is the task/problem?
   - Why is it relevant and difficult?

2. **Data & Metrics**
   - Dataset details (e.g., size, duration, labels, format)
   - What are the task’s evaluation metrics?

3. **Baseline Method**
   - Technical breakdown of the official DCASE baseline model for the task

4. **Two SOTA Approaches**
   - Choose and review two submitted methods  
   - For each method:  
     - Motivate your selection  
     - Explain its novelty and performance  
     - Provide a **technical overview** of all major components  
   - ⚠️ Only choose methods with **available open-source implementations**

5. **Project Plan**
   - Choose **one** of your two SOTA methods  
   - Propose **two major technical modifications** (details below)

---

## 🔧 Modification Requirements

You must propose and justify two substantial modifications to your selected SOTA method:

### 1. Signal Processing Front-End Change
Examples:
- Replace Mel Spectrum with Scattering Transform  
- Modify input layers of the neural network accordingly

### 2. Neural Network Architecture Change
Examples:
- Replace GRU with Transformer  
- Adapt model pipeline (e.g., loss function, training regime) accordingly

> **❗ Insufficient Modifications Include:**
> - Switching to log-frequency spectrogram
> - Adding extra CNN layers
> - Swapping output layers (e.g. softmax to SVM)

### 📌 For Each Modification:
- Justify **why** it’s a good idea  
- Describe **how** you plan to implement it  
- List expected **benefits or trade-offs**  
- Outline key **(hyper-)parameters** to experiment with and why

---

## 🗓️ Key Dates

- **April 1 (Before 6pm)**:  
  - Submit your slide deck (PDF) to **Gradescope**  
  - Recommended: use **Google Slides** to create your deck

- **April 1 (Class Time)**:
  - **15-minute presentation**, plus **5-minute Q&A**  
  - All group members must present **technical content**  
  - Arrive **10+ minutes early** to test your setup  
  - Late arrival will affect your grade

- **March 31 (Optional)**:
  - Special **office hours** to discuss project proposals with Juan or Julia  
  - Details TBA

---

## 📦 Final Deliverables (Due May 6)

You will submit:

- 📂 A **GitHub repository** with your implementation  
- 📝 A **written report** (details will be shared later)  
- 📽️ A **final presentation and slides**

---

## 💡 Tips & Resources

- Explore DCASE tasks and results:
  - https://dcase.community/challenge2024
- Use the **results page** to find:
  - Baseline models
  - Submitted approaches
  - Technical reports
  - External publications via citations

---

Happy Listening! 🎧  

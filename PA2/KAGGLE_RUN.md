# Minimal Kaggle steps (you)

Code and tests are already done locally. Training needs Kaggle T4x2 GPUs.

## One-time

1. Open https://www.kaggle.com while logged in as **contactrifacsg@gmail.com**
2. **Account → Create New API Token** → downloads `kaggle.json`  
   (optional if you only use the website UI)
3. Create notebook: **New Notebook** → Accelerator **GPU T4 x2** → Internet **On**

## Run training (website UI)

1. File → Import notebook → select  
   `D:\LUMS\CS5326 Advanced GenAI and agents\Assignments\PA2\notebooks\kaggle_end_to_end.ipynb`  
   **or** paste cells from that file.
2. Confirm the clone URL is  
   `https://github.com/RifaSalman/CS5326.git`  
   and set the working directory to the `PA2` subdirectory after cloning.
3. Run all cells. Budget ~8–9 hours end-to-end.
4. After **SFT**, **each DPO**, and **generation**: **Save Version** so outputs persist.
5. Download `/kaggle/working/pa2_outputs.zip` (or attach notebook output as a dataset).

## Judging

1. New notebook configured as a **utility benchmark** / with model access.
2. Attach the generation outputs dataset.
3. Run the judge cells / `scripts/judge_pairwise_kaggle.py` with `--reuse_judgments`.

## After Kaggle finishes

Copy `outputs/` and figures back into this PA2 folder, then tell me — I will fill the report tables and export the PDF.

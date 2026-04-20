# OMNI-vocal-pipeline

**End-to-end AI Vocal Cloning Pipeline**  
Full song → Automatic Lead/Backup vocal separation → RVC cloning with **maevn1** model → Prompt-based AI Mastering

Optimized for **Google Colab Free Tier (T4 GPU)**.

### Features
- Generic `new_render_slot` — just drop any audio files
- Efficient lead vs backup vocals separation (T4-friendly models)
- RVC inference with your `maevn1.pth`
- Prompt-driven AI mastering ("loud streaming master", "warm vinyl", "punchy club", etc.)
- Fully modular and easy to maintain

### Quick Start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maevnholdingtrust/OMNI-vocal-pipeline/blob/main/main_pipeline.ipynb)

1. Click the badge above to open in Colab
2. Run Cell 1 (Setup)
3. Run Cell 2 (Upload your `maevn1.pth` + index)
4. Drop your songs into the `new_render_slot` folder
5. Run the processing + mastering cells
6. Download your mastered maevn1 clones

### Repository Structure
- `main_pipeline.ipynb` → Complete single notebook
- `requirements.txt` → All dependencies
- `README.md` → This file

### Future Plans
- Separate Python modules
- Gradio web UI
- Mix cloned vocal with instrumental
- Better reference-based mastering

---

Made with ❤️ for maevn1 voice cloning

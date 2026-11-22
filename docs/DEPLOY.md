# Deploying the app and the model

## 1. Put the model on Hugging Face (once per new model)
1. Train in `notebooks/train_on_colab.ipynb` (T4 GPU).
2. In Colab, open 🔑 **Secrets** → add `HF_TOKEN` = your Hugging Face **write** token → enable notebook access.
3. Run the "Export + upload" cells. They export ONNX, check accuracy, and upload
   `model.onnx` + a model card to `https://huggingface.co/<you>/drone-flight-plan-fields`.
4. If your Hugging Face username is not `yuvendhan`, change `hf_repo` in `src/drone_flight_plan/config.py`
   (or type it in the app under *More options*).

The app downloads the model automatically on first start (cached afterwards). No PyTorch needed to run it.

## 2a. Live demo on Hugging Face Spaces (free)
1. huggingface.co → **New Space** → name `drone-flight-plan` → SDK **Docker** → Blank → Create.
2. Upload the two files from `deploy/hf-space/` (`README.md`, `Dockerfile`) to the Space (Files → Add file).
3. Wait for the build (a few minutes). Your demo: `https://huggingface.co/spaces/<you>/drone-flight-plan`.
4. After pushing new code to GitHub: Space → Settings → **Factory rebuild**.

## 2b. Or: Streamlit Community Cloud (free)
1. share.streamlit.io → **New app** → pick the GitHub repo, branch `main`, file `app/streamlit_app.py`.
2. Advanced settings → Python 3.11 → Deploy. It installs from `requirements.txt`.

## Notes
- "Use my location" needs **https** (both hosts provide it) or `localhost`.
- Place search uses OpenStreetMap Nominatim (light use only); satellite tiles are Esri World Imagery
  (personal testing / demo only — use your own drone imagery for real work).

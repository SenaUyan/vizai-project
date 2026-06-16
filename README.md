# VizAI

VizAI is an agentic AI system for automated exploratory data analysis on CSV datasets.

It profiles uploaded CSV files, selects a primary analysis target, generates focused visualizations, produces textual insights, and evaluates analysis quality.

## Features
- CSV upload
- dataset profiling
- primary target detection
- automatic chart selection
- insight generation
- evaluation scoring

## Files
- `app.py`: Streamlit interface
- `agent.py`: decision-making logic
- `tools.py`: data profiling and visualization tools

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
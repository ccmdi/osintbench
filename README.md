OSINTbench is a benchmark for evaluating how well large language models can perform open-source intelligence (OSINT) tasks. Categories include:
* **Geolocation**: Spatial reasoning
* **Identification**: Information synthesis, breadth of knowledge
* **Temporal**: Temporal reasoning
* **Analysis**: General reasoning

# **[Leaderboard](https://osintbench.org)**

# Installation
```bash
git clone https://github.com/ccmdi/osintbench.git
cd osintbench
pip install -r requirements.txt
```

Setup your `.env` based on `SAMPLE.env` for whichever model providers you wish to test for (e.g. `ANTHROPIC_API_KEY` must be set to test Claude).

**You will need to manually create a dataset for this to work**. Datasets follow this schema:
```json
"cases": [
    {
      "id": <case_number>,
      "images": [
        "images/<image_number>.<ext>"
      ],
      "info": "<context given to the model about the case>",
      "tasks": [
        {
          "id": 1,
          "type": "location",
          "prompt": "Find the exact location of the photo.",
          "answer": {
            "lat": <true_lat>,
            "lng": <true_lng>
          }
        },
        {
            "id": 2,
            "type": "identification",
            "prompt": "Who is this?",
            "answer": "<person_name>"
        }
      ]
    },
    ...
```

The folder for a dataset should be in the structure:
```
dataset/
├─ basic/
│  ├─ metadata.json
│  ├─ images/
│  │  ├─ 2.jpg
│  │  ├─ 1.png
├─ advanced/
│  ├─ metadata.json
```
Where your dataset definition lives in `metadata.json`.

## Test a model
> [!CAUTION]
> Most outputs are evaluated by a judge model. Double-check responses before finalizing results.

```
python osintbench.py --dataset <test name> --model <model name>
```

Models go by their class name in `models.py`. Gemini 2.5 Flash goes by `Gemini2_5Flash`, for instance.

# Roadmap
- [x] **Tool use**
    - [x] Google Search
    - [x] EXIF extraction
    - [x] Reverse image search (Google Lens)
    - [x] Visit website
    - [x] Overpass turbo
    - [x] Google Street View
- [ ] **High quality, human-verified datasets**
- [ ] Higher prompt quality to improve performance
- [ ] Video support?
- [ ] Recursive prompting/self-evaluation
- [ ] Release

> [!NOTE]
> Contributors are welcome! Check the roadmap.
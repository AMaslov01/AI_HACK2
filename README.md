# Task: AI Agent for Scientific Literature Analysis

You need to implement an **AI agent** powered by [GigaChat-2-Max](https://developers.sber.ru/docs/ru/gigachat/models/gigachat-2-max) that, given the materials of a scientific paper (LaTeX source, PDF, illustrations), answers questions about the paper — including questions that require **understanding figures and diagrams**.

---

## 1. Data

### 1.1. Format

- **Paper**: a folder containing **LaTeX source files** (`TeX source`) **and** a **PDF version** of the paper.
- The **PDF** is provided for participants who prefer to use it in their solution and for development purposes; during scoring, the `data/` folder will contain **both** the TeX source and the PDF.
- The source folder may contain:
  - one or more `*.tex` files with the main text and/or sections;
  - auxiliary files: `.cls`, `.sty`, `.bst`, `.bib`, etc.;
  - images: `png`, `jpg`, `pdf` (all PDF files inside the `TeX source` folder are **single-page** images);
  - nested **subfolders** with **arbitrary** names.

### 1.2. Constraints (important for design)

- The folder **structure** is **not** fixed: one paper may have a single `*.tex` file, another may have many files and subfolders;
- The **full text** of the paper is **longer** than the model's context window;
- For **parsing/processing LaTeX** to split and prepare the text, use **`langchain_text_splitters`**;

### 1.3. Example `TeX source` Folder Contents

<details>
<summary>Example 1 (many files in root + images)</summary>

```text
.
├── 00README.json
├── B1937+21.png
├── bhb_merger_rate_models.pdf
├── cimento.cls
├── corner_NANOGrav.pdf
├── …
├── new-main.tex
├── references.bib
└── …
```

</details>

<details>
<summary>Example 2 (figures subfolder, multiple .tex files)</summary>

```text
.
├── 00README.json
├── feature_learning.tex
├── figures
│   ├── C_xx.pdf
│   ├── depth_scales.png
│   └── …
├── macros.tex
└── stat_mech_nn_…_lecture.tex
```

</details>

<details>
<summary>Example 3 (nearly a single main .tex)</summary>

```text
.
├── 00README.json
├── JHEP.bst
├── Strong-CP-Lecture-v2.bbl
├── Strong-CP-Lecture-v2.tex
└── tmp.txt
```

</details>

### 1.4. Debug Examples

- You are provided with papers for developing and debugging your agent:
  1. [Paper #1](https://disk.yandex.ru/d/YiZBjU8xVxg88w)
  2. [Paper #2](https://disk.yandex.ru/d/mS1__DMT4D2yng)
- You may use any other papers if they help make the agent more robust across different formats and domains.

---

## 2. Input and Output

| | |
|---|---|
| **Input** | Paper materials in `data/`: the paper PDF, **subfolder(s)** `TeX source`, and the file **`data/questions.txt`** — a numbered list of questions. |
| **Output** | The file **`output/answers.txt`** — **numbered** answers **in the same order** as the questions. |

**Important:**

1. The format of `answers.txt` is fixed (see examples in the provided papers): for example, the answer to question 3 must begin with the line `## Answer 3`, followed by the answer text;

2. It is likely worth imposing a time limit on how long the agent spends on each question;

3. If the agent skips any question, it must still write the line `## Answer ...` into `answers.txt`, with the answer body containing, for example, `no answer`.

---

## 3. Agent Requirements

- Must answer **all** questions about the paper's content, **including** questions that require **analysis of images/diagrams**;
- **Launch**: a single command from the repository root: `python run.py`;
- **Model**: **GigaChat-2-Max only**;
- **Evaluation mode**: the agent operates **autonomously**, in a **closed** environment, **without** participant interaction during the run;
- **Confidentiality and competition ethics**: it is **forbidden** to transmit **any** content from the paper, questions, or answers **outside** the environment (external APIs, public chats, arbitrary outgoing requests with task source materials). Violation leads to **disqualification**.

---

## 4. Permitted Libraries

Libraries for the virtual environment are listed in `pyproject.toml`.

---

## 5. Environment Setup

```bash
uv venv
uv sync
```

Only two parameters need to be specified in `.env`:

```bash
GIGACHAT_CREDENTIALS='<Token provided by the organizers>'
GIGACHAT_SCOPE='GIGACHAT_API_CORP'
```

`.env` **must** be included in the submitted zip archive.

---

## 6. Self-Check (Locally)

After running `python run.py`, execute `src/utils/check_submission.py`. The script checks for the presence of `.env`, the presence of `output/answers.txt`, and that the number of answers matches the number of questions.

```bash
python run.py
python src/utils/check_submission.py
```

---

## 7. Submission

You submit a **zip archive with code** that will be run on the [evaluation platform](http://risk-hackathon.ru).

Your code must:
- run in a clean environment;
- read data independently from the `data/` folder;
- save results to the `output/` folder;
- Maximum agent runtime: **15 minutes per paper**.

The submission limit applies only to successful attempts (number of successful attempts = 1).

The platform will verify that answers were produced; answer quality will be evaluated separately, outside the platform.

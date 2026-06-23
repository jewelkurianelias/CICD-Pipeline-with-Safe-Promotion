# CI/CD Pipeline with Safe Promotion and Rollback

This project demonstrates Continuous Integration (CI) and Continuous Delivery (CD) for Machine Learning models. Instead of pushing models manually, you will use an automated "Gate" that mathematically proves a new model is better before letting it reach production. If disaster strikes, you have an instant Rollback script.

## 📁 Project Structure

```text
├── .github/
│   └── workflows/
│       └── ci_cd.yml       # GitHub Actions CI/CD pipeline configuration
├── data/                   # (Generated) Synthetic training and holdout data
├── src/
│   ├── generate_data.py    # Generates synthetic demand data
│   ├── train.py            # Trains and registers the candidate model
│   ├── evaluate.py         # CD Gate comparing candidate vs. production
│   └── rollback.py         # Emergency rollback procedure
├── tests/
│   └── test_pipeline.py    # Unit tests run during CI
├── README.md               # Project documentation
├── requirements.txt        # Python package dependencies
└── deployment_log.txt      # (Generated) Auditable log of model deployments
```

## 🔄 The Pipeline Flow

* **CI (Continuous Integration):** When code is pushed, `pytest` ensures the pipeline logic isn't broken.
* **Train:** The pipeline trains a model and tags it as a `@candidate`.
* **CD (Continuous Delivery) Gate:** `evaluate.py` compares the `@candidate` against the live `@production` model on a hidden holdout dataset. It only promotes the candidate if the Mean Absolute Error (MAE) is strictly better.

## 🚀 Quickstart: Local Simulation

You don't need a real GitHub account to test this! We will manually run the steps that the GitHub Actions YAML file (in `.github/workflows/`) runs automatically.

**1. Set up the environment:**

```bash
python -m venv venv
# Mac/Linux: source venv/bin/activate
# Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the CI Tests:**

```bash
python -m pytest tests/
```

**3. Train the initial model (Version 1):**

```bash
python src/train.py
python src/evaluate.py
```

> **Result:** Because there is no production model yet, Version 1 will be automatically promoted to `@production`.

## 🛑 Test the CD Gate (Fail a Bad Model)

Let's pretend a data scientist accidentally pushed terrible code. Let's see how the pipeline protects itself.

**1. Train a deliberately bad model:**

```bash
python src/train.py --degrade
```

> **Note:** This generates Version 2 and tags it as the new `@candidate`.

**2. Trigger the CD Gate:**

```bash
python src/evaluate.py
```

> **Result:** The script will print 🛑 **REJECTED**. It proves Version 2 is worse than Version 1, blocking it from production!

## ⏪ Test the Rollback Procedure

If a bad model somehow gets promoted (e.g., data drift or a bug in the gate), you need to revert immediately without waiting 20 minutes for a new model to train.

**1. First, let's force a bad model into production so we have something to fix:**

```bash
python src/train.py --degrade
python src/evaluate.py  # (This will reject it)
# Force the bad model into production manually to simulate an outage:
python -c "import mlflow; mlflow.tracking.MlflowClient().set_registered_model_alias('DemandForecaster', 'production', '2')"
```

**2. Execute the Rollback:**

```bash
python src/rollback.py
```

> **Result:** In less than a second, the script shifts the `@production` alias back to Version 1. The outage is mitigated instantly.

**3. Check the deployment log:**

```bash
cat deployment_log.txt
```

## ☁️ Running in GitHub Actions (The Real World)

To see this pipeline run automatically in the cloud, exactly as a real DevOps team would:

1.  **Create a GitHub Repository:** Go to GitHub.com and create a new, empty repository.
2.  **Push Your Code:** In your terminal, initialize Git and push this project to your new repository:

```bash
git init
git add .
git commit -m "Initial commit with ML CI/CD pipeline"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

3.  **Watch the Magic:** Go to your repository on GitHub and click the **Actions** tab at the top.
    * You will instantly see a job called "ML CI/CD Pipeline" running.
    * Click on it to watch GitHub automatically spin up a cloud server, install your dependencies, run your unit tests, train the model, and execute the CD evaluation gate—all completely hands-free!

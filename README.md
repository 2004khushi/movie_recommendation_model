# 🎬 Movie Recommendation System (Content-Based, Unsupervised)

This project implements a **content-based movie recommendation system** using **unsupervised learning techniques**.  
Movies are recommended based on semantic similarity of their content and refined using filtering and ranking logic.

---

## 📌 Project Overview

The system recommends movies similar to a given movie by:
- analyzing textual metadata (overview and tags)
- computing semantic similarity using TF-IDF and cosine similarity
- applying logical filters and ranking signals to improve relevance

The system does **not rely on user ratings as labels** and does not use collaborative filtering.

---

## 🧠 Methodology

### 1. Data Cleaning
- Removed duplicate movie entries to ensure index consistency
- Reset dataframe index after cleaning
- Handled missing values safely without introducing semantic bias

---

### 2. Feature Engineering

**Similarity Features (Core ML):**
- Movie overview
- Combined tags (genre, keywords, etc.)

**Filtering / Constraints:**
- Language (`original_language`, `spoken_languages`)
- Runtime
- Release status
- Adult content flag

**Ranking Signals:**
- Vote average
- Vote count (log-scaled)
- Popularity

Each feature group is used strictly for its intended role.

---

### 3. Vectorization
- Text features are vectorized using **TF-IDF**
- N-grams are enabled to capture phrase-level semantics
- Stopwords are removed
- Resulting vectors are sparse and high-dimensional

---

### 4. Similarity Computation
- **Cosine similarity** is used to measure semantic closeness
- For a given movie, top-K similar candidates are retrieved

---

### 5. Filtering & Ranking
After similarity:
- Movies are filtered based on availability, safety, language, and runtime
- Remaining candidates are ranked using a weighted score combining:
  - similarity
  - rating quality
  - confidence (vote count)
  - popularity

This avoids feature leakage and keeps the system modular.

---

## 🏗️ System Pipeline

```text
Input Movie Title
        ↓
TF-IDF Vector Lookup
        ↓
Cosine Similarity (Top-K Candidates)
        ↓
Filtering (Language, Runtime, Safety)
        ↓
Ranking (Votes, Popularity)
        ↓
Final Recommendations
```

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## 📂 Project Structure

- movies.ipynb – Complete notebook containing:
   - data preprocessing
   - feature engineering
   - similarity computation
   - filtering & ranking logic
- Serialized artifacts (for deployment-ready use):
    - TF-IDF vectorizer
    - TF-IDF matrix
    - Movie metadata
    - Title-to-index mapping
 
## 🎯 Key Learnings

- Practical handling of unsupervised ML systems
- Feature role separation (similarity vs filtering vs ranking)
- Sparse vector representations and similarity search
- Real-world debugging with pandas and sklearn
- Designing ML systems beyond single-algorithm solutions

# 🎬 Movie Recommendation System (Content-Based)

This project implements a **content-based movie recommendation system** using **unsupervised learning techniques**.  
Movies are recommended based on semantic similarity of their content rather than user ratings or collaborative signals.

---

## 📌 Project Overview

The system recommends movies similar to a given movie by:
- analyzing textual metadata (overview, tags)
- computing semantic similarity using TF-IDF and cosine similarity
- applying logical filters and ranking signals to refine recommendations

Unlike collaborative filtering, this system **does not rely on user interaction data** and works purely on movie attributes.

---

## 🧠 Approach & Methodology

### 1. Feature Engineering
The following features are used:

**Similarity Features (Core ML):**
- Movie overview
- Tags (combined textual metadata)

**Filtering / Constraints:**
- Language (`original_language`, `spoken_languages`)
- Runtime
- Release status
- Adult content flag

**Ranking Signals (Post-similarity):**
- Vote average
- Vote count (log-scaled)
- Popularity

---

### 2. Vectorization
- Text data is vectorized using **TF-IDF**
- N-grams are enabled to capture phrase-level semantics
- Stopwords are removed
- Resulting vectors are high-dimensional and sparse

---

### 3. Similarity Computation
- **Cosine similarity** is used to measure semantic closeness between movies
- For a given movie, the top-K most similar candidates are retrieved

---

### 4. Filtering & Ranking
After similarity:
- Movies are filtered based on availability, safety, language, and runtime
- Remaining candidates are ranked using a weighted scoring function combining:
  - similarity score
  - rating quality
  - confidence (vote count)
  - popularity

This separation ensures clean system design and avoids feature leakage.

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

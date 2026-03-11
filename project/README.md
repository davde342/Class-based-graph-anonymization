# DPP: Graph Anonymization Project

## Giusto Davide, 5252480

This project include an implementation of class-based graph anonymization algorithms for interaction graphs and the possibility to conduct different types of analysis on how they work on a movies dataset.

## Overview

**Key Implementation:**
- **Label List Anonymization** (Full and Prefix patterns)
- **Partitioning Anonymization**
- **Privacy vs. Utility Analysis**
- **Statistical Impact Analysis**
- **Efficiency Analysis**
- **Correctness Analysis**

## Quick Start

1. **Start the CLI**:
```bash
python cli.py
```

2. **Load a dataset** (Command 1):
   - Enter path to ratings.csv (or use default: `ml-latest-small/ratings.csv`)
   - Enter path to movies.csv (or use default: `ml-latest-small/movies.csv`)
   - Set minimum ratings per user (default: 5)

3. **Set parameters** (Command 2):
   - Set `m` (max class size)
   - Set `k` (prefix size for Label List)
   - Choose sort order to use in the creation of classes (default: num_ratings)

4. **Run anonymization** (Commands 3-6):
   - Try different methods to see their effects

5. **Test with queries** (Commands 8-10):
   - Compare results between original and anonymized graphs

## CLI Usage Guide

The CLI provides a menu with the following main sections:

### Data Loading (Commands 1-2)

#### Command 1: Load MovieLens Dataset

#### Command 2: Set Anonymization Parameters

**Parameters Explanation:**
- **m**: Maximum number of nodes in each anonymization class
- **k**: For Label List prefix pattern, number of elements in each label list (k ≤ m)
- **sort_order**: Attributes to sort nodes by (available: age, num_ratings, group)

### Anonymization (Commands 3-7)

#### Command 3: Label List (Full Pattern)
Creates label lists where each node gets the complete list of all nodes in its class.

#### Command 4: Label List (Prefix Pattern)
Creates label lists of size k using a cycling pattern through the class.

#### Command 5: Partitioning
Groups nodes into classes and publishes interaction counts per class.

#### Command 6: Run All Methods
Executes all three anonymization methods sequentially.

#### Command 7: View Anonymized Graphs
Displays sample output from anonymized graphs (first 10 entries).

### Queries (Commands 8-10)

#### Command 8: Genre-Based Query
```
Minimum ratings threshold (default 50): 100
Target genre (default 'Sci-Fi'): Action
```
Finds users with high activity who rated movies of a specific genre.

#### Command 9: Degree Distribution Query
```
Target number of ratings (default 45): 50
```
Finds users with exactly the specified number of ratings.

#### Command 10: Structural Similarity Query
```
Minimum shared movies (default 5): 10
```
Finds user pairs sharing at least the specified number of movies.

### Analysis (Commands 11-14)

#### Command 11: Privacy vs. Utility Analysis
Runs analysis on how different parameter values affect the privacy-utility tradeoff. Results saved to `reduced_utility_analysis/` directory.

#### Command 12: Statistical Analysis
Analyzes how anonymization affects statistical properties like average age per genre. Results saved to `genre_age_analysis/` directory.

#### Command 13: Efficiency Analysis
Measure execution time of different anonymization methods. Results saved to `efficiency_analysis/` directory.

#### Command 14: Algorithm Correctness Validation
Validates that all algorithms work correctly and maintain safety conditions.

### Utilities (Command 15)

#### Command 15: Export Data
```
Output directory (default ./output): ./my_results
```
Exports anonymized graphs to JSON files for external analysis.

## Anonymization Methods

### 1. Label List (Full Pattern)
- **Mechanism**: Each node in a class gets identical label list containing all class members

### 2. Label List (Prefix Pattern)
- **Mechanism**: Each node gets a label list of size k using cycling pattern

### 3. Partitioning
- **Mechanism**: Publishes aggregated interaction counts per class


## File Structure

```
project/
├── cli.py                   # Main CLI interface
├── __init__.py              # Package initialization
├── 
├── # Core Algorithm Files
├── label_list.py            # Label List anonymization
├── partitioning.py          # Partitioning anonymization
├── utils.py                 # Utility functions and safety conditions
├── 
├── # Query and Analysis
├── query.py                 # Query implementations
├── privacy_utility_level.py # Privacy vs utility analysis
├── statistical_analysis.py # Statistical impact analysis
├── efficiency_analysis.py   # Performance benchmarking
├── algorithm_correctness.py # Validation tools
├── 
├── # Dataset
├── ml-latest-small/         # MovieLens dataset
│   ├── ratings.csv
│   ├── movies.csv
│   └── ...
├── 
└── # Results (generated during execution)
    ├── detailed_comparison/     # Detailed analysis results
    ├── efficiency_analysis/     # Performance benchmarks
    ├── genre_age_analysis/      # Statistical analysis results
    ├── reduced_utility_analysis/ # Privacy-utility analysis
    └── statistical_analysis/   # General statistics
```

## Dataset Requirements

### Primary: MovieLens Dataset
The system is designed to work with MovieLens datasets. Download from [GroupLens](https://grouplens.org/datasets/movielens/).

**Required files:**
- `ratings.csv`: User-movie ratings (userId, movieId, rating, timestamp)
- `movies.csv`: Movie metadata (movieId, title, genres)

### Custom Datasets
For custom datasets, ensure your data follows this structure:

**Ratings format:**
```csv
userId,movieId,rating,timestamp
1,31,2.5,1260759144
1,1029,3.0,1260759179
```

**Movies format:**
```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

## References

Based on research paper Class-based graph anonymization for social network data

Smriti BhagatRutgers University

Graham CormodeAT&T Labs–Research

Balachander KrishnamurthyAT&T Labs–Research

Divesh SrivastavaAT&T Labs–Research


---



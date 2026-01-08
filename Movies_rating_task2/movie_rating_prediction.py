# ==============================
# movie_rating_analysis.py - FIXED VERSION
# Complete Solution for Movie Rating Prediction & Analysis
# ==============================

# ==============================
# 1. DATA CLEANING MODULE
# ==============================
def clean_movie_data(input_file='IMDb Movies India.csv', output_file='cleaned_movies_data.csv'):
    """
    Clean the raw movie data and save to a new CSV file
    """
    import pandas as pd
    import numpy as np
    import re
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 80)
    print("STEP 1: DATA CLEANING")
    print("=" * 80)
    
    # Try different encodings to read the file
    encodings = ['latin1', 'utf-8', 'ISO-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully loaded {input_file} with {encoding} encoding")
            break
        except:
            continue
    
    if df is None:
        try:
            df = pd.read_csv(input_file)
            print(f"Successfully loaded {input_file} with default encoding")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # 1. Clean Name column
    print("\n1. Cleaning Name column...")
    df_clean['Name'] = df_clean['Name'].fillna('Unknown Movie')
    # Remove extra whitespace
    df_clean['Name'] = df_clean['Name'].str.strip()
    
    # 2. Clean Year column
    print("2. Cleaning Year column...")
    def extract_year(year_str):
        if pd.isna(year_str):
            return np.nan
        year_str = str(year_str)
        # Extract year from parentheses or any format
        matches = re.findall(r'\b(19\d{2}|20\d{2})\b', year_str)
        if matches:
            try:
                return int(matches[-1])  # Take the last match if multiple
            except:
                return np.nan
        # Try to extract any 4-digit number
        matches = re.findall(r'\b\d{4}\b', year_str)
        if matches:
            try:
                year = int(matches[-1])
                if 1900 <= year <= 2024:  # Reasonable year range
                    return year
            except:
                pass
        return np.nan
    
    df_clean['Year'] = df_clean['Year'].apply(extract_year)
    
    # 3. Clean Duration column
    print("3. Cleaning Duration column...")
    def extract_duration(duration_str):
        if pd.isna(duration_str):
            return np.nan
        duration_str = str(duration_str)
        # Extract first set of digits
        matches = re.findall(r'\b(\d+)\b', duration_str)
        if matches:
            try:
                duration = int(matches[0])
                if 10 <= duration <= 400:  # Reasonable movie duration range
                    return duration
            except:
                pass
        return np.nan
    
    df_clean['Duration'] = df_clean['Duration'].apply(extract_duration)
    
    # 4. Clean Genre column
    print("4. Cleaning Genre column...")
    df_clean['Genre'] = df_clean['Genre'].fillna('Unknown')
    df_clean['Genre'] = df_clean['Genre'].str.strip()
    # Extract primary genre (first one if multiple)
    df_clean['Primary_Genre'] = df_clean['Genre'].apply(
        lambda x: str(x).split(',')[0].strip() if ',' in str(x) else str(x).strip()
    )
    
    # 5. Clean Rating column
    print("5. Cleaning Rating column...")
    df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
    
    # 6. Clean Votes column
    print("6. Cleaning Votes column...")
    def clean_votes(votes_str):
        if pd.isna(votes_str):
            return np.nan
        votes_str = str(votes_str)
        # Remove commas and any non-numeric characters
        votes_str = re.sub(r'[^\d.]', '', votes_str)
        if votes_str:
            try:
                return float(votes_str)
            except:
                return np.nan
        return np.nan
    
    df_clean['Votes'] = df_clean['Votes'].apply(clean_votes)
    
    # 7. Clean Director column
    print("7. Cleaning Director column...")
    df_clean['Director'] = df_clean['Director'].fillna('Unknown')
    df_clean['Director'] = df_clean['Director'].str.strip()
    
    # 8. Clean Actor columns
    print("8. Cleaning Actor columns...")
    for col in ['Actor 1', 'Actor 2', 'Actor 3']:
        df_clean[col] = df_clean[col].fillna('Unknown')
        df_clean[col] = df_clean[col].str.strip()
    
    # 9. Handle missing values systematically
    print("9. Handling missing values...")
    
    # Fill Year with median based on decade if possible
    if df_clean['Year'].notna().any():
        year_median = df_clean['Year'].median()
        df_clean['Year'].fillna(year_median, inplace=True)
    else:
        df_clean['Year'].fillna(2000, inplace=True)  # Default year
    
    # Convert Year to integer
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    # Fill Duration with median
    if df_clean['Duration'].notna().any():
        duration_median = df_clean['Duration'].median()
        df_clean['Duration'].fillna(duration_median, inplace=True)
    else:
        df_clean['Duration'].fillna(120, inplace=True)  # Default 2 hours
    
    # Fill Votes with median
    if df_clean['Votes'].notna().any():
        votes_median = df_clean['Votes'].median()
        df_clean['Votes'].fillna(votes_median, inplace=True)
    else:
        df_clean['Votes'].fillna(50, inplace=True)
    
    # Fill Rating with median based on Year and Genre
    if df_clean['Rating'].notna().any():
        # Calculate median rating by Year and Genre
        rating_medians = df_clean.groupby(['Year', 'Primary_Genre'])['Rating'].median()
        
        def fill_rating(row):
            if pd.isna(row['Rating']):
                key = (row['Year'], row['Primary_Genre'])
                if key in rating_medians:
                    return rating_medians[key]
                else:
                    # Try just by Year
                    year_median = df_clean[df_clean['Year'] == row['Year']]['Rating'].median()
                    if not pd.isna(year_median):
                        return year_median
                    else:
                        return df_clean['Rating'].median()
            return row['Rating']
        
        df_clean['Rating'] = df_clean.apply(fill_rating, axis=1)
        # Fill any remaining NaNs with overall median
        df_clean['Rating'].fillna(df_clean['Rating'].median(), inplace=True)
    else:
        df_clean['Rating'].fillna(5.0, inplace=True)  # Default rating
    
    # 10. Remove duplicates
    print("10. Removing duplicates...")
    initial_count = len(df_clean)
    # Remove exact duplicates
    df_clean = df_clean.drop_duplicates()
    # Remove duplicates based on Name and Year (same movie released in same year)
    df_clean = df_clean.drop_duplicates(subset=['Name', 'Year'], keep='first')
    final_count = len(df_clean)
    duplicates_removed = initial_count - final_count
    print(f"   Removed {duplicates_removed} duplicate entries")
    
    # 11. Create additional features
    print("11. Creating additional features...")
    
    # Create decade feature
    df_clean['Decade'] = (df_clean['Year'] // 10) * 10
    
    # Create movie age (years since release)
    current_year = 2024
    df_clean['Movie_Age'] = current_year - df_clean['Year']
    
    # Create success categories
    df_clean['Success_Category'] = pd.cut(df_clean['Rating'], 
                                          bins=[0, 4, 6, 7.5, 10],
                                          labels=['Flop', 'Average', 'Good', 'Excellent'])
    
    # Create popularity score
    df_clean['Popularity_Score'] = df_clean['Rating'] * np.log1p(df_clean['Votes'])
    
    # Create genre count
    df_clean['Genre_Count'] = df_clean['Genre'].apply(
        lambda x: len(str(x).split(',')) if ',' in str(x) else 1
    )
    
    # 12. Remove unrealistic data
    print("12. Removing unrealistic data...")
    initial_count = len(df_clean)
    
    # Remove movies with unrealistic years
    df_clean = df_clean[(df_clean['Year'] >= 1900) & (df_clean['Year'] <= 2024)]
    
    # Remove movies with unrealistic durations
    df_clean = df_clean[(df_clean['Duration'] >= 30) & (df_clean['Duration'] <= 300)]
    
    # Remove movies with unrealistic ratings
    df_clean = df_clean[(df_clean['Rating'] >= 1) & (df_clean['Rating'] <= 10)]
    
    final_count = len(df_clean)
    invalid_removed = initial_count - final_count
    print(f"   Removed {invalid_removed} entries with unrealistic values")
    
    # 13. Sort data
    df_clean = df_clean.sort_values(['Year', 'Rating'], ascending=[False, False])
    
    # 14. Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # 15. Save cleaned data
    print("13. Saving cleaned data...")
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nCleaning completed!")
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Cleaned data saved to: {output_file}")
    
    print(f"\nMissing values after cleaning:")
    print(df_clean.isnull().sum())
    
    print(f"\nSample of cleaned data:")
    print(df_clean[['Name', 'Year', 'Duration', 'Primary_Genre', 'Rating', 'Votes']].head())
    
    return df_clean

# ==============================
# 2. EXPLORATORY DATA ANALYSIS MODULE
# ==============================
def perform_eda(df_clean):
    """
    Perform exploratory data analysis on cleaned data
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Set visualization style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    
    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 40)
    
    numeric_cols = ['Year', 'Duration', 'Rating', 'Votes', 'Movie_Age', 'Popularity_Score']
    numeric_stats = df_clean[numeric_cols].describe()
    print(numeric_stats)
    
    # 2. Data Distribution Analysis
    print("\n2. DATA DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Rating Distribution
    axes[0, 0].hist(df_clean['Rating'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df_clean['Rating'].mean(), color='red', linestyle='--', 
                      label=f"Mean: {df_clean['Rating'].mean():.2f}")
    axes[0, 0].axvline(df_clean['Rating'].median(), color='green', linestyle='--',
                      label=f"Median: {df_clean['Rating'].median():.2f}")
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Movie Ratings')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Duration Distribution
    axes[0, 1].hist(df_clean['Duration'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df_clean['Duration'].mean(), color='red', linestyle='--',
                      label=f"Mean: {df_clean['Duration'].mean():.1f} min")
    axes[0, 1].set_xlabel('Duration (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Movie Duration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Votes Distribution (log scale)
    axes[0, 2].hist(np.log1p(df_clean['Votes']), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('log(Votes + 1)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Votes (Log Scale)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Year Distribution
    axes[1, 0].hist(df_clean['Year'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Release Years')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success Categories
    success_counts = df_clean['Success_Category'].value_counts()
    axes[1, 1].pie(success_counts.values, labels=success_counts.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Movie Success Categories')
    
    # Genre Distribution (Top 10)
    top_genres = df_clean['Primary_Genre'].value_counts().head(10)
    axes[1, 2].barh(range(len(top_genres)), top_genres.values)
    axes[1, 2].set_yticks(range(len(top_genres)))
    axes[1, 2].set_yticklabels(top_genres.index)
    axes[1, 2].set_xlabel('Number of Movies')
    axes[1, 2].set_title('Top 10 Movie Genres')
    axes[1, 2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Trend Analysis
    print("\n3. TREND ANALYSIS OVER TIME")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Movies per year
    movies_per_year = df_clean.groupby('Year').size()
    axes[0, 0].plot(movies_per_year.index, movies_per_year.values, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Movies')
    axes[0, 0].set_title('Movies Released Per Year')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average rating per year
    avg_rating_per_year = df_clean.groupby('Year')['Rating'].mean()
    axes[0, 1].plot(avg_rating_per_year.index, avg_rating_per_year.values, 
                   marker='o', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Rating')
    axes[0, 1].set_title('Average Movie Rating Per Year')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average duration per year
    avg_duration_per_year = df_clean.groupby('Year')['Duration'].mean()
    axes[1, 0].plot(avg_duration_per_year.index, avg_duration_per_year.values,
                   marker='o', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Average Duration (minutes)')
    axes[1, 0].set_title('Average Movie Duration Per Year')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Votes per year (log scale)
    avg_votes_per_year = df_clean.groupby('Year')['Votes'].mean()
    axes[1, 1].plot(avg_votes_per_year.index, np.log1p(avg_votes_per_year.values),
                   marker='o', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('log(Average Votes + 1)')
    axes[1, 1].set_title('Average Votes Per Year (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trends_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Correlation Analysis
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Calculate correlations
    correlation_matrix = df_clean[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Detailed Analysis
    print("\n5. DETAILED ANALYSIS")
    print("-" * 40)
    
    # Year with best average rating (with at least 10 movies)
    yearly_stats = df_clean.groupby('Year').agg({
        'Rating': ['mean', 'count'],
        'Votes': 'mean',
        'Duration': 'mean'
    }).round(2)
    
    yearly_stats.columns = ['Avg_Rating', 'Movie_Count', 'Avg_Votes', 'Avg_Duration']
    qualified_years = yearly_stats[yearly_stats['Movie_Count'] >= 10]
    
    if not qualified_years.empty:
        best_year = qualified_years['Avg_Rating'].idxmax()
        best_year_rating = qualified_years['Avg_Rating'].max()
        print(f"Year with best average rating (≥10 movies): {best_year} ({best_year_rating:.2f})")
    else:
        best_year = yearly_stats['Avg_Rating'].idxmax()
        best_year_rating = yearly_stats['Avg_Rating'].max()
        print(f"Year with best average rating: {best_year} ({best_year_rating:.2f})")
    
    # Impact of duration on rating
    duration_bins = [0, 90, 120, 150, 180, 300]
    duration_labels = ['<90min', '90-120min', '120-150min', '150-180min', '>180min']
    df_clean['Duration_Category'] = pd.cut(df_clean['Duration'], 
                                          bins=duration_bins, 
                                          labels=duration_labels)
    
    duration_rating_stats = df_clean.groupby('Duration_Category')['Rating'].agg(['mean', 'count']).round(2)
    print(f"\nAverage Rating by Duration Categories:")
    print(duration_rating_stats)
    
    # Genre analysis
    genre_stats = df_clean.groupby('Primary_Genre').agg({
        'Rating': ['mean', 'count'],
        'Votes': 'mean'
    }).round(2)
    genre_stats.columns = ['Avg_Rating', 'Movie_Count', 'Avg_Votes']
    
    # Top genres by rating (with at least 10 movies)
    qualified_genres = genre_stats[genre_stats['Movie_Count'] >= 10]
    if not qualified_genres.empty:
        top_genres_by_rating = qualified_genres.sort_values('Avg_Rating', ascending=False).head(5)
        print(f"\nTop 5 Genres by Average Rating (≥10 movies):")
        print(top_genres_by_rating)
    
    # Top genres by number of movies
    top_genres_by_count = genre_stats.sort_values('Movie_Count', ascending=False).head(5)
    print(f"\nTop 5 Most Common Genres:")
    print(top_genres_by_count)
    
    # Director analysis
    director_stats = df_clean.groupby('Director').agg({
        'Rating': ['mean', 'count'],
        'Votes': 'mean'
    }).round(2)
    director_stats.columns = ['Avg_Rating', 'Movie_Count', 'Avg_Votes']
    
    # Top directors by rating (with at least 5 movies)
    qualified_directors = director_stats[director_stats['Movie_Count'] >= 5]
    if not qualified_directors.empty:
        top_directors = qualified_directors.sort_values('Avg_Rating', ascending=False).head(5)
        print(f"\nTop 5 Directors by Average Rating (≥5 movies):")
        print(top_directors)
    
    # Director with most movies
    most_prolific_director = director_stats['Movie_Count'].idxmax()
    most_prolific_count = director_stats['Movie_Count'].max()
    print(f"\nMost Prolific Director: {most_prolific_director} ({most_prolific_count} movies)")
    
    # Actor analysis
    all_actors = pd.concat([df_clean['Actor 1'], df_clean['Actor 2'], df_clean['Actor 3']])
    actor_counts = all_actors.value_counts()
    
    # Actor in most movies (excluding 'Unknown')
    if 'Unknown' in actor_counts:
        actor_counts = actor_counts.drop('Unknown')
    if not actor_counts.empty:
        most_frequent_actor = actor_counts.idxmax()
        most_frequent_count = actor_counts.max()
        print(f"\nActor in Most Movies: {most_frequent_actor} ({most_frequent_count} movies)")
    
    # Top movies analysis
    print(f"\nTop 10 Highest Rated Movies (with at least 100 votes):")
    top_movies = df_clean[df_clean['Votes'] >= 100].sort_values(['Rating', 'Votes'], 
                                                                ascending=[False, False]).head(10)
    for i, (idx, row) in enumerate(top_movies.iterrows(), 1):
        print(f"{i:2}. {row['Name'][:40]:40} ({row['Year']}): {row['Rating']:.1f} "
              f"(Votes: {int(row['Votes']):,}, Genre: {row['Primary_Genre']})")
    
    return {
        'numeric_stats': numeric_stats,
        'best_year': best_year,
        'best_year_rating': best_year_rating,
        'duration_stats': duration_rating_stats,
        'top_genres_by_rating': top_genres_by_rating,
        'most_prolific_director': most_prolific_director,
        'most_frequent_actor': most_frequent_actor
    }

# ==============================
# 3. FEATURE ENGINEERING MODULE - FIXED
# ==============================
def prepare_features_for_modeling(df_clean):
    """
    Prepare features for machine learning model
    """
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    # Create a copy for feature engineering
    df_features = df_clean.copy()
    
    # 1. Director features
    print("1. Creating director features...")
    
    # Director experience (number of movies directed)
    director_counts = df_features['Director'].value_counts()
    df_features['Director_Experience'] = df_features['Director'].map(director_counts)
    
    # Director average rating
    director_avg_rating = df_features.groupby('Director')['Rating'].mean()
    df_features['Director_Avg_Rating'] = df_features['Director'].map(director_avg_rating)
    
    # Fill NaN values
    df_features['Director_Experience'].fillna(1, inplace=True)
    df_features['Director_Avg_Rating'].fillna(df_features['Rating'].mean(), inplace=True)
    
    # 2. Actor features
    print("2. Creating actor features...")
    
    # Combine all actors
    all_actors = pd.concat([df_features['Actor 1'], 
                           df_features['Actor 2'], 
                           df_features['Actor 3']])
    
    # Actor popularity (number of movies)
    actor_counts = all_actors.value_counts()
    
    # Actor average rating
    actor_ratings = {}
    for actor in actor_counts.index:
        if actor != 'Unknown':
            actor_movies = df_features[(df_features['Actor 1'] == actor) | 
                                      (df_features['Actor 2'] == actor) | 
                                      (df_features['Actor 3'] == actor)]
            if len(actor_movies) > 0:
                actor_ratings[actor] = actor_movies['Rating'].mean()
    
    # Map actor features to movies
    df_features['Actor1_Experience'] = df_features['Actor 1'].map(actor_counts)
    df_features['Actor2_Experience'] = df_features['Actor 2'].map(actor_counts)
    df_features['Actor3_Experience'] = df_features['Actor 3'].map(actor_counts)
    
    df_features['Actor1_Avg_Rating'] = df_features['Actor 1'].map(actor_ratings)
    df_features['Actor2_Avg_Rating'] = df_features['Actor 2'].map(actor_ratings)
    df_features['Actor3_Avg_Rating'] = df_features['Actor 3'].map(actor_ratings)
    
    # Fill NaN values
    for col in ['Actor1_Experience', 'Actor2_Experience', 'Actor3_Experience']:
        df_features[col].fillna(1, inplace=True)
    
    overall_avg_rating = df_features['Rating'].mean()
    for col in ['Actor1_Avg_Rating', 'Actor2_Avg_Rating', 'Actor3_Avg_Rating']:
        df_features[col].fillna(overall_avg_rating, inplace=True)
    
    # Calculate average actor metrics
    df_features['Avg_Actor_Experience'] = df_features[['Actor1_Experience', 
                                                      'Actor2_Experience', 
                                                      'Actor3_Experience']].mean(axis=1)
    
    df_features['Avg_Actor_Rating'] = df_features[['Actor1_Avg_Rating',
                                                  'Actor2_Avg_Rating',
                                                  'Actor3_Avg_Rating']].mean(axis=1)
    
    # 3. Genre features - FIXED: Use existing Genre_Count
    print("3. Creating genre features...")
    
    # Use existing Genre_Count as Genre_Diversity
    df_features['Genre_Diversity'] = df_features['Genre_Count']
    
    # One-hot encode top genres
    top_genres = df_features['Primary_Genre'].value_counts().head(10).index
    for genre in top_genres:
        col_name = f'Genre_{genre.replace(" ", "_").replace("/", "_")[:20]}'
        df_features[col_name] = (df_features['Primary_Genre'] == genre).astype(int)
    
    # 4. Temporal features - FIXED: Use existing Decade and add missing ones
    print("4. Creating temporal features...")
    
    # Ensure all decades are represented
    all_decades = list(range(1910, 2030, 10))  # From 1910 to 2020
    for decade in all_decades:
        col_name = f'Decade_{decade}'
        df_features[col_name] = (df_features['Decade'] == decade).astype(int)
    
    # Seasonality features
    df_features['Is_Recent'] = (df_features['Year'] >= 2010).astype(int)
    
    # 5. Other features
    print("5. Creating other features...")
    
    # Has famous director
    df_features['Has_Famous_Director'] = (df_features['Director_Experience'] > 5).astype(int)
    
    # Has famous actor
    df_features['Has_Famous_Actor'] = (
        (df_features['Actor1_Experience'] > 5) |
        (df_features['Actor2_Experience'] > 5) |
        (df_features['Actor3_Experience'] > 5)
    ).astype(int)
    
    # Success indicator
    df_features['Is_Successful'] = (df_features['Rating'] >= 7).astype(int)
    
    # 6. Select features for modeling - FIXED: Ensure all features exist
    print("6. Selecting features for modeling...")
    
    # Base features - only those that definitely exist
    base_features = [
        'Year', 'Duration', 'Genre_Diversity', 'Movie_Age',
        'Director_Experience', 'Director_Avg_Rating',
        'Avg_Actor_Experience', 'Avg_Actor_Rating',
        'Has_Famous_Director', 'Has_Famous_Actor', 'Is_Recent'
    ]
    
    # Add genre dummies (check they exist)
    genre_features = [col for col in df_features.columns if col.startswith('Genre_') and col in df_features.columns]
    
    # Add decade dummies (check they exist)
    decade_features = [col for col in df_features.columns if col.startswith('Decade_') and col in df_features.columns]
    
    # Combine all features
    feature_columns = base_features + genre_features + decade_features
    
    # Verify all features exist
    missing_features = [f for f in feature_columns if f not in df_features.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing: {missing_features[:5]}...")
        # Remove missing features
        feature_columns = [f for f in feature_columns if f in df_features.columns]
    
    # Target variable
    target_column = 'Rating'
    
    print(f"Total features created: {len(feature_columns)}")
    print(f"Total samples for modeling: {len(df_features)}")
    
    # Return the modified dataframe with all features
    return df_features, feature_columns, target_column

# ==============================
# 4. MACHINE LEARNING MODULE
# ==============================
def build_prediction_model(df_features, feature_columns, target_column):
    """
    Build machine learning models for rating prediction
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    import joblib
    
    print("\n" + "=" * 80)
    print("STEP 4: MACHINE LEARNING MODELING")
    print("=" * 80)
    
    # Check if we have enough data
    if len(df_features) < 100:
        print(f"Warning: Only {len(df_features)} samples available. Model may not be reliable.")
        print("Consider collecting more data for better predictions.")
    
    # Prepare data
    X = df_features[feature_columns].fillna(df_features[feature_columns].mean())
    y = df_features[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=5, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std
            }
            
            print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
            print(f"  5-Fold CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
            
        except Exception as e:
            print(f"  Error training {name}: {str(e)[:100]}...")
    
    # Display results
    if results:
        results_df = pd.DataFrame(results).T
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(results_df.round(4))
        
        # Find best model
        best_model_name = results_df['Test_R2'].idxmax()
        best_model = trained_models[best_model_name]
        best_results = results_df.loc[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Test R²: {best_results['Test_R2']:.4f}")
        print(f"Test RMSE: {best_results['Test_RMSE']:.4f}")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            print("\n" + "=" * 80)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("=" * 80)
            
            # Get feature importance
            importance_values = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': importance_values
            }).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            top_features = feature_importance.head(20)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(top_features)), top_features['Importance'].values)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top 20 Feature Importance - {best_model_name}')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Top 10 Most Important Features:")
            print(top_features.head(10).to_string(index=False))
        
        # Prediction visualization
        print("\n" + "=" * 80)
        print("PREDICTION VISUALIZATION")
        print("=" * 80)
        
        y_pred = best_model.predict(X_test)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Rating')
        axes[0].set_ylabel('Predicted Rating')
        axes[0].set_title(f'Actual vs Predicted\n{best_model_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Rating')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Error distribution
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(residuals.mean(), color='r', linestyle='--',
                       label=f'Mean: {residuals.mean():.3f}')
        axes[2].set_xlabel('Prediction Error')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Prediction Errors')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save the best model
        model_filename = 'best_movie_rating_predictor.pkl'
        joblib.dump(best_model, model_filename)
        print(f"\nBest model saved as: {model_filename}")
        
        # Save feature columns
        feature_filename = 'model_features.pkl'
        joblib.dump(feature_columns, feature_filename)
        print(f"Feature columns saved as: {feature_filename}")
        
        return best_model, feature_columns, results_df, df_features
    
    else:
        print("No models were successfully trained.")
        return None, None, None, df_features

# ==============================
# 5. PREDICTION & REPORTING MODULE - FIXED
# ==============================
def generate_predictions_and_report(df_with_features, model, feature_columns, analysis_results):
    """
    Generate predictions and create final report
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    print("\n" + "=" * 80)
    print("STEP 5: PREDICTIONS & FINAL REPORT")
    print("=" * 80)
    
    # 1. Make predictions on entire dataset
    print("\n1. Generating predictions...")
    
    if model is not None and feature_columns is not None:
        # Prepare features for prediction
        # Ensure all feature columns exist in the dataframe
        available_features = [f for f in feature_columns if f in df_with_features.columns]
        missing_features = [f for f in feature_columns if f not in df_with_features.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing from dataframe")
            print(f"Missing: {missing_features[:5]}...")
        
        if available_features:
            X_all = df_with_features[available_features].fillna(df_with_features[available_features].mean())
            
            # Predict ratings
            df_with_features['Predicted_Rating'] = model.predict(X_all)
            
            # Calculate prediction error
            df_with_features['Prediction_Error'] = df_with_features['Rating'] - df_with_features['Predicted_Rating']
            df_with_features['Absolute_Error'] = np.abs(df_with_features['Prediction_Error'])
            
            print(f"Average prediction error: {df_with_features['Prediction_Error'].mean():.3f}")
            print(f"Average absolute error: {df_with_features['Absolute_Error'].mean():.3f}")
            
            # Find movies with largest prediction errors
            print("\nTop 5 Movies with largest prediction errors:")
            large_errors = df_with_features.nlargest(5, 'Absolute_Error')[['Name', 'Year', 'Rating', 
                                                                          'Predicted_Rating', 
                                                                          'Absolute_Error']]
            print(large_errors.round(3).to_string())
    else:
        print("No model available for predictions.")
        df_with_features['Predicted_Rating'] = df_with_features['Rating']
        df_with_features['Prediction_Error'] = 0
        df_with_features['Absolute_Error'] = 0
    
    # 2. Generate insights
    print("\n2. Key Insights:")
    print("-" * 40)
    
    insights = []
    
    # Year analysis
    current_year = datetime.now().year
    recent_movies = df_with_features[df_with_features['Year'] >= current_year - 10]
    
    if len(recent_movies) > 0:
        avg_recent_rating = recent_movies['Rating'].mean()
        older_movies = df_with_features[df_with_features['Year'] < current_year - 10]
        
        if len(older_movies) > 0:
            avg_older_rating = older_movies['Rating'].mean()
            
            insights.append(f"• Recent movies (last 10 years) have an average rating of {avg_recent_rating:.2f}")
            insights.append(f"• Older movies have an average rating of {avg_older_rating:.2f}")
            
            if avg_recent_rating > avg_older_rating:
                insights.append("• Recent movies are rated higher than older movies")
            else:
                insights.append("• Older movies are rated higher than recent movies")
    
    # Genre insights
    if 'Primary_Genre' in df_with_features.columns:
        genre_stats = df_with_features.groupby('Primary_Genre').agg({
            'Rating': ['mean', 'count']
        }).round(2)
        genre_stats.columns = ['Avg_Rating', 'Count']
        
        top_rated_genres = genre_stats[genre_stats['Count'] >= 10].nlargest(3, 'Avg_Rating')
        if len(top_rated_genres) > 0:
            genre_list = ', '.join([f'{g} ({r:.2f})' for g, r in zip(top_rated_genres.index, top_rated_genres['Avg_Rating'])])
            insights.append(f"• Top rated genres: {genre_list}")
    
    # Duration insights
    if 'Duration' in df_with_features.columns:
        # Create duration bins
        duration_bins = [0, 90, 120, 150, 180, 300]
        duration_labels = ['<90min', '90-120min', '120-150min', '150-180min', '>180min']
        
        df_with_features['Duration_Group'] = pd.cut(df_with_features['Duration'], 
                                                   bins=duration_bins, 
                                                   labels=duration_labels)
        
        best_duration = df_with_features.groupby('Duration_Group')['Rating'].mean().idxmax()
        insights.append(f"• Highest rated movies are typically {best_duration} long")
    
    # Director insights
    if 'most_prolific_director' in analysis_results:
        insights.append(f"• Most prolific director: {analysis_results['most_prolific_director']}")
    
    # Print insights
    for insight in insights:
        print(insight)
    
    # 3. Create final visual report
    print("\n3. Creating final visual report...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top genres by rating and count
    if 'Primary_Genre' in df_with_features.columns:
        top_genres_by_rating = df_with_features.groupby('Primary_Genre')['Rating'].mean().nlargest(5)
        axes[0, 0].bar(range(len(top_genres_by_rating)), top_genres_by_rating.values)
        axes[0, 0].set_xticks(range(len(top_genres_by_rating)))
        axes[0, 0].set_xticklabels(top_genres_by_rating.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].set_title('Top 5 Genres by Average Rating')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Genre Data', ha='center', va='center')
        axes[0, 0].set_title('Top 5 Genres by Average Rating')
    
    # Rating distribution by decade
    if 'Decade' in df_with_features.columns:
        rating_by_decade = df_with_features.groupby('Decade')['Rating'].mean()
        axes[0, 1].plot(rating_by_decade.index, rating_by_decade.values, marker='o', linewidth=2)
        axes[0, 1].set_xlabel('Decade')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].set_title('Average Rating by Decade')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Decade Data', ha='center', va='center')
        axes[0, 1].set_title('Average Rating by Decade')
    
    # Success rate over time
    if 'Is_Successful' in df_with_features.columns:
        success_rate_by_year = df_with_features.groupby('Year')['Is_Successful'].mean()
        axes[1, 0].plot(success_rate_by_year.index, success_rate_by_year.values * 100, 
                       marker='o', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Movie Success Rate Over Time (Rating ≥ 7)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Success Data', ha='center', va='center')
        axes[1, 0].set_title('Movie Success Rate Over Time')
    
    # Director experience vs rating
    if 'Director_Experience' in df_with_features.columns:
        sample_size = min(1000, len(df_with_features))
        sample = df_with_features.sample(sample_size, random_state=42)
        axes[1, 1].scatter(sample['Director_Experience'], sample['Rating'], 
                          alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Director Experience (Number of Movies)')
        axes[1, 1].set_ylabel('Rating')
        axes[1, 1].set_title('Director Experience vs Movie Rating')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Director Experience Data', ha='center', va='center')
        axes[1, 1].set_title('Director Experience vs Movie Rating')
    
    plt.tight_layout()
    plt.savefig('final_insights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Save final dataset with predictions
    output_file = 'movies_with_predictions.csv'
    df_with_features.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nFinal dataset with predictions saved as: {output_file}")
    
    # 5. Generate summary report
    print("\n4. Generating summary report...")
    
    report = []
    report.append("=" * 80)
    report.append("MOVIE RATING ANALYSIS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nDATASET OVERVIEW:")
    report.append(f"  • Total movies analyzed: {len(df_with_features):,}")
    report.append(f"  • Year range: {df_with_features['Year'].min()} - {df_with_features['Year'].max()}")
    report.append(f"  • Average rating: {df_with_features['Rating'].mean():.2f}")
    report.append(f"  • Average duration: {df_with_features['Duration'].mean():.1f} minutes")
    
    report.append(f"\nKEY FINDINGS:")
    if 'best_year' in analysis_results:
        report.append(f"  • Best year for ratings: {analysis_results['best_year']} "
                     f"(Rating: {analysis_results['best_year_rating']:.2f})")
    
    if 'most_prolific_director' in analysis_results:
        report.append(f"  • Most prolific director: {analysis_results['most_prolific_director']}")
    
    if 'most_frequent_actor' in analysis_results:
        report.append(f"  • Actor in most movies: {analysis_results['most_frequent_actor']}")
    
    report.append(f"\nTRENDS:")
    if 'Primary_Genre' in df_with_features.columns:
        most_common_genre = df_with_features['Primary_Genre'].mode()
        if not most_common_genre.empty:
            report.append(f"  • Most common genre: {most_common_genre.iloc[0]}")
    
    report.append(f"  • Average votes per movie: {df_with_features['Votes'].mean():,.0f}")
    
    if 'model_r2' in analysis_results:
        report.append(f"\nPREDICTION MODEL:")
        report.append(f"  • Best model: Random Forest")
        report.append(f"  • Test R² score: {analysis_results['model_r2']:.3f}")
        report.append(f"  • Prediction error (MAE): {df_with_features['Absolute_Error'].mean():.3f}")
    
    report.append(f"\nRECOMMENDATIONS:")
    report.append("  1. Focus on genres that consistently receive higher ratings")
    report.append("  2. Consider director experience when planning productions")
    report.append("  3. Optimal movie duration is around 120-150 minutes")
    report.append("  4. Cast experienced actors for better rating potential")
    report.append("  5. Release timing can impact movie success")
    
    report.append(f"\n" + "=" * 80)
    
    # Save report to file
    report_filename = 'analysis_summary_report.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    print(f"\nFull report saved as: {report_filename}")
    
    return df_with_features

# ==============================
# 6. MAIN EXECUTION FUNCTION - FIXED
# ==============================
def main():
    """
    Main function to execute the complete analysis pipeline
    """
    print("=" * 80)
    print("MOVIE RATING PREDICTION & ANALYSIS SYSTEM")
    print("=" * 80)
    
    try:
        # Step 1: Clean the data
        df_clean = clean_movie_data('IMDb Movies India.csv', 'cleaned_movies_data.csv')
        
        if df_clean is None or len(df_clean) == 0:
            print("Error: Could not load or clean data. Exiting.")
            return
        
        # Step 2: Perform EDA
        analysis_results = perform_eda(df_clean)
        
        # Step 3: Prepare features for modeling
        # This returns df_features which has all the engineered features
        df_features, feature_columns, target_column = prepare_features_for_modeling(df_clean)
        
        # Step 4: Build prediction model
        if len(df_features) >= 50:  # Only build model if we have enough data
            model, features, model_results, df_with_all_features = build_prediction_model(
                df_features, feature_columns, target_column
            )
            
            if model_results is not None:
                # Add model results to analysis results
                best_model_name = model_results['Test_R2'].idxmax()
                analysis_results['model_r2'] = model_results.loc[best_model_name, 'Test_R2']
                analysis_results['model_rmse'] = model_results.loc[best_model_name, 'Test_RMSE']
        else:
            print(f"\nInsufficient data for modeling. Only {len(df_features)} samples available.")
            model = None
            features = None
            df_with_all_features = df_features  # Use the features dataframe
        
        # Step 5: Generate predictions and final report
        # Use df_with_all_features which has all the engineered features
        final_df = generate_predictions_and_report(
            df_with_all_features, 
            model, 
            feature_columns if features is None else features,
            analysis_results
        )
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("1. cleaned_movies_data.csv - Cleaned dataset")
        print("2. movies_with_predictions.csv - Dataset with predictions")
        print("3. data_distributions.png - Data distribution visualizations")
        print("4. trends_over_time.png - Trend analysis visualizations")
        print("5. correlation_matrix.png - Correlation heatmap")
        
        if model is not None:
            print("6. feature_importance.png - Feature importance plot")
            print("7. prediction_analysis.png - Model performance visualizations")
            print("8. best_movie_rating_predictor.pkl - Trained model")
            print("9. model_features.pkl - Model features")
        
        print("10. final_insights.png - Final insights visualization")
        print("11. analysis_summary_report.txt - Complete analysis report")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

# ==============================
# 7. EXECUTE THE PIPELINE
# ==============================
if __name__ == "__main__":
    # Install required packages if not already installed
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'xgboost',
        'scipy'
    ]
    
    print("Checking for required packages...")
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages are installed. Starting analysis...\n")
    
    # Run the main analysis pipeline
    main()
🎬 Netflix Movies Analysis
A data analysis project on Netflix-style movie data, exploring trends in genres, popularity, votes, and more using Python, Pandas, and Seaborn.

📁 Dataset
The dataset contains 9,800+ movie records with the following fields:

Title

Release Date

Genre(s)

Popularity Score

Vote Count & Average

Original Language

Poster URL

🧹 Data Cleaning
Converted Release_Date to year format

Removed unnecessary columns (Overview, Original_Language, Poster_Url)

Categorized Vote_Average into:

not_popular, below_avg, average, popular

Split multi-genre values and exploded into separate rows

Handled outliers and dropped NaN values

📊 Exploratory Data Analysis
Key Questions Answered:
Most Frequent Genre?

Drama is the most frequent genre (over 14% of entries)

Genres by Popularity/Vote Category?

Bar plots show distribution of movies across vote categories

Most/Least Popular Movies?

📈 Most Popular: Spider-Man: No Way Home

📉 Least Popular: The United States vs. Billie Holiday, Threads

Most Productive Release Year?

Histogram shows distribution of movies by year

📉 Visualizations
Genre distribution bar chart

Vote category distribution

Popularity extremes by title and genre

Release trends over time

🛠 Tools Used
Python (Pandas, NumPy)

Matplotlib

Seaborn

▶️ How to Run
This project is available to run in Google Colab and jupyter notebook

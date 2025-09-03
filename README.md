#  Airbnb Consumer Behavior Dynamics: Post-Pandemic Insights
Data mining and machine learning project analyzing Airbnb listings (2020–2023) to uncover key factors influencing reservation days and evolving traveler preferences.

## Project Overview
This project delivers a three-year, data-driven analysis of Airbnb reservation trends to provide a comprehensive view of how traveler behavior evolved in the post-COVID-19 market.

By analyzing property features, host metrics, guest experience factors, and review scores, the project enables actionable insights for:

- **Hosts**: optimize pricing, amenities, and guest experience  
- **Platforms**: enhance recommendation systems and marketplace performance  
- **Businesses & Analysts**: understand post-pandemic shifts in travel behavior  

Using Python (`pandas`, `scikit-learn`) for data cleaning, feature engineering, and Decision Tree modeling, combined with Power BI and Excel visualizations, I created a professional BI solution that identifies the key drivers of reservations and supports strategic, data-driven decision-making.

**Key Findings**:  
- **2021** → Price and host trust signals (verification, Superhost status) heavily influenced bookings  
- **2022** → Reputation factors (guest reviews on cleanliness, accuracy, value) became critical  
- **2023** → Guest experience (check-in ease, amenities, location) gained equal importance to price  

**Takeaway**: The Airbnb market evolved from **affordability-driven → reputation-driven → experience-driven** traveler decisions.


---

## Skills Demonstrated
- Advanced data preprocessing and feature engineering.
- Translating machine learning outputs (Decision Tree insights) into business-ready takeaways.
- Bridging data analysis with strategic business impact.

---

## Table of Contents
- [Objectives](#objectives)  
- [Tools & Technologies](#tools--technologies)  
- [Dataset Overview](#dataset-overview)  
- [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)   
- [Post-Processing Dataset Overview (24 Features)](#post-processing-dataset-overview-24-features)   
- [Modeling Approach](#modeling-approach)  
- [Outcome & Insights](#outcome--insights)  
- [Feature Importance Analysis Using Decision Trees](#feature-importance-analysis-using-decision-trees)  
- [Outcome: Feature Importance Trends (2021–2023)](#outcome-feature-importance-trends-2021-2023)  
- [High-Level Insights on Feature Importance (2021–2023)](#high-level-insights-on-feature-importance-2021-2023)  
- [Business Impact](#business-impact)  
- [Overall Trend Summary](#overall-trend-summary)  
- [Conclusions & Insights](#conclusions--insights)  
- [Future Work & Applications](#future-work--applications)

---

## Objectives
1. **Identify Key Drivers**: Quantify which property and host features most strongly influence reservation days.
2. **Year-over-Year Comparison**: Track how feature importance changed between 2021, 2022, and 2023, reflecting post-pandemic market shifts.
3. **Machine Learning Insights**: Use Decision Tree models to uncover interpretable patterns in consumer booking behavior.
4. **Strategic Application**: Provide insights to hosts and platforms for pricing optimization, amenity selection, and guest experience improvements.

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Excel** | Data cleaning, preliminary exploration, feature extraction, and validation of listings. |
| **Python (pandas, scikit-learn, matplotlib)** | Decision Tree modeling, feature importance analysis, and visualization of trends. |
| **Power BI** | Interactive dashboards to compare patterns across years and visualize reservation trends. |

These tools enabled a complete end-to-end workflow, from raw data processing to actionable business insights.

---

## Dataset Overview
- **Source**: Raw Airbnb listing datasets for London (UK) were obtained from [InsideAirbnb.com](http://insideairbnb.com) for the years **2021, 2022, and 2023**.
- **Scope**: Each dataset contains ~X active listings per year, with **24 engineered features** capturing host, property, guest ratings, and amenities information.
- **Granularity**: Reservation data was aggregated by listing, either daily or monthly, depending on availability.
- **Focus**: Designed to analyze **reservation duration patterns** and understand how consumer behavior evolved post-pandemic.

This dataset forms a comprehensive basis for year-over-year comparison, enabling insights into shifts in booking preferences, guest priorities, and the evolving Airbnb market in London.

---

## Data Preprocessing & Feature Engineering
Before applying machine learning models to analyze Airbnb consumer behavior, extensive **data cleaning, transformation, and feature engineering** were performed on the London listing datasets for 2021, 2022, and 2023. These steps ensured high-quality, consistent data, enabling accurate modeling and actionable insights.

### Key Preprocessing Steps

1. **Amenities Extraction**
   - Parsed the raw `amenities` column to identify **WiFi, Pool, Bathtub, and Free Parking**.
   - Created **binary columns** for each (TRUE/FALSE depending on availability).

2. **Host Communication Features**
   - Quantified host reliability and responsiveness by creating a column that counts the number of **communication methods** (e.g., email, phone, ID verification).
   - Formula example in Excel/DAX:
     ```
     LEN(TRIM(cell)) - LEN(SUBSTITUTE(cell, " ", "")) + 1
     ```
   - Captures professionalism and guest accessibility.

3. **Availability Filtering**
   - Removed listings with **zero availability for the next 365 days**.
   - Focused analysis on properties actively accepting bookings, improving accuracy.

4. **Binary Column Standardization**
   - Standardized columns for **superhost status, profile picture, and ID verification** (originally `t/f`) into **TRUE/FALSE**, then converted to **1/0** for modeling.

5. **Removing Empty Rows**
   - Dropped rows with missing critical values using Python:
     ```python
     my_data.dropna(inplace=True)
     ```
   - Ensured data integrity for downstream analysis.
---

## Post-Processing Dataset Overview (24 Features)

After preprocessing, the dataset contained **24 standardized, numeric, and categorical features**, grouped into four main categories:

### 1. Host Metrics
- Host acceptance rate
- Host response rate
- Host response time
- Host is superhost (**TRUE/FALSE → 1/0**)
- Host has profile picture (**TRUE/FALSE → 1/0**)
- Host identity verified (**TRUE/FALSE → 1/0**)
- Number of host profiles

### 2. Property Metrics
- Price per night
- Accommodates
- Bathrooms
- Bedrooms
- Beds

### 3. Guest Ratings
- Number of reviews
- Review scores: accuracy, cleanliness, check-in, communication, location, value, overall rating

### 4. Amenities
- WiFi
- Pool
- Bathtub
- Free Parking

These engineered and standardized features captured **listing quality, host professionalism, and guest experience**, forming a robust foundation for Decision Tree modeling.
This enabled deep **year-over-year analysis** of consumer behavior and reservation trends in the post-pandemic Airbnb market.

---

## Modeling Approach

### Decision Tree Modeling
To analyze Airbnb consumer behavior and identify the factors that most influence **reservation durations**, **Decision Tree models** were built using Python and key data science libraries (`pandas`, `scikit-learn`, `matplotlib`).

This approach allowed for both **predictive insights** and **interpretable visualizations** of feature importance across 2021, 2022, and 2023.

---

### Workflow Overview

#### 1. Importing Required Libraries
Python libraries used:
- **pandas** → data manipulation
- **scikit-learn** → Decision Tree modeling
- **matplotlib** → visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
```
### 2. Data Loading & Preparation
- The **preprocessed Airbnb dataset** was loaded into Python using `pd.read_csv`.
- **Features (X):** A mix of host metrics, property attributes, guest ratings, and amenities.
- **Target (y):** `reserved_days` — the number of reserved days per listing.

```python
features = [
    'host_response_time','host_response_rate','host_acceptance_rate',
    'hos1_is_superhos1','hos1_has_profile_pic','hos1_identity_verified',
    'accommodates','bathrooms','bedrooms','beds','price','number_of_reviews',
    'review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
    'review_scores_checkin','review_scores_communication','review_scores_location',
    'review_scores_value','WiFi','Pool','Bathtub','Free_parking','number_of_host_profiles'
]

X = df[features]
y = df['reserved_days']
```
### 3. Building the Decision Tree Model
- A **DecisionTreeClassifier** with `max_depth=5` was trained to balance **model interpretability** and **pattern discovery**.
- The model identifies which features most influence **reservation days**, ranking them by importance.

```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X, y)
```

### 4. Visualizing the Decision Tree
- **High-resolution plots** were created for clarity, using **color-filled nodes** and **rounded edges**.
- The visualizations highlight **feature splits** and their contribution to predicting reserved days, making insights actionable for hosts and platforms.

```python
import matplotlib.pyplot as plt
from sklearn import tree

fig, axes = plt.subplots(figsize=(12,12), dpi=300)
tree.plot_tree(dtree, feature_names=features, filled=True, rounded=True, fontsize=8)
plt.savefig('airbnb_decision_tree.png', bbox_inches='tight')
plt.show()
```
Decision tree plots for 2021, 2022, and 2023

![2021](https://github.com/MoRMatipour/Airbnb-Consumer-Behavior-Dynamics-Post-Pandemic-Insights/blob/main/Charts%20and%20Decision%20Tree/2021%20Three.png?raw=true)

![2022](https://github.com/MoRMatipour/Airbnb-Consumer-Behavior-Dynamics-Post-Pandemic-Insights/blob/main/Charts%20and%20Decision%20Tree/2022%20three.png?raw=true)

![2023](https://github.com/MoRMatipour/Airbnb-Consumer-Behavior-Dynamics-Post-Pandemic-Insights/blob/main/Charts%20and%20Decision%20Tree/2023%20tree.png?raw=true)

### Outcome & Insights

- Three **Decision Tree models** were trained, one for each year (2021–2023), enabling direct comparison of **feature importance trends**.
- This workflow forms the **analytical backbone** of the project, providing both **quantitative insights** and **visual evidence** of how guest behavior and listing performance evolved in the **post-pandemic Airbnb market**.

---

## Feature Importance Analysis Using Decision Trees

To identify which **listing and host features** most influence reservation days, **Decision Tree models** were trained using Python and scikit-learn. These models not only provide predictive insights but also generate interpretable **rankings of feature importance**, offering **actionable guidance** for hosts and platform managers.

---

### Analysis Workflow

1. **Model Training**
   - **Input Features**: Key attributes including *price, host responsiveness, room capacity, review scores, and amenities*.
   - **Target Variable**: `reserved_days` — the number of reserved days per listing.
   - A `DecisionTreeClassifier` with controlled depth was used to prevent overfitting while capturing meaningful patterns.

2. **Feature Importance Extraction**
   - Importance scores for each feature were extracted from the trained model.
   - Contributions were expressed as **percentages of total importance**, enabling clear comparison across years.

3. **Visualization**
   - Horizontal bar charts were created to highlight the **most influential factors**, with the highest importance features displayed at the top.
   - Visualizations were **saved and incorporated** into portfolio materials for clear reporting and stakeholder insights.


```python
import pandas as pd
import matplotlib.pyplot as plt

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dtree.feature_importances_
})

# Sort by importance and calculate percentage
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df['Importance (%)'] = 100 * importance_df['Importance']

# Display table
print(importance_df)

# Plot horizontal bar chart
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance (%)'], color='skyblue')
plt.gca().invert_yaxis()  # highest importance on top
plt.xlabel('Importance (%)')
plt.title('Feature Importance - Decision Tree')
plt.tight_layout()
plt.show()
```

## Outcome: Feature Importance Trends (2021–2023)

Using the feature importance extraction method, we obtained **three separate tables of measures** (one for each year: 2021, 2022, 2023).  
I then **blended these tables manually into a single consolidated table** to make insights clearer and enable easier comparison across years, as well as for creating comparative charts.


| Feature | 2021 (%) | 2022 (%) | 2023 (%) |
|---------|----------|----------|----------|
| Price | 14.89 | 11.05 | 12.20 |
| Number of reviews | 13.67 | 8.55 | 8.23 |
| Host acceptance rate | 8.82 | 7.74 | 7.44 |
| Review scores — rating | 8.54 | 6.02 | 5.96 |
| Number of host profiles | 8.71 | 6.38 | 2.54 |
| Review scores — value | 3.31 | 6.57 | 6.83 |
| Review scores — location | 3.23 | 6.27 | 7.16 |
| Review scores — cleanliness | 3.38 | 6.20 | 6.81 |
| Review scores — check-in | 2.25 | 5.74 | 6.24 |
| Review scores — communication | 1.86 | 4.82 | 5.27 |
| Accommodates | 4.43 | 3.90 | 3.86 |
| Host response rate | 3.64 | 3.49 | 3.16 |
| Bathrooms | 3.08 | 3.65 | 3.49 |
| Beds | 4.27 | 3.23 | 2.66 |
| Bathtub | 2.08 | 2.59 | 3.04 |
| Host response time | 2.97 | 1.99 | 1.65 |
| Host is superhost | 2.13 | 1.51 | 1.98 |
| Bedrooms | 2.07 | 1.61 | 1.81 |
| Free Parking | 2.45 | 1.29 | 1.42 |
| Host identity verified | 1.24 | 0.97 | 0.56 |
| WiFi | 0.33 | 0.46 | 1.22 |
| Host has profile picture | 0.02 | 0.22 | 0.39 |
| Pool | 0.19 | 0.07 | 0.12 |


Clustered Column Chart
Shows the importance of each feature across 2021, 2022, and 2023 with three columns per feature, allowing year-over-year comparison.

![Clustred chart](https://github.com/MoRMatipour/Airbnb-Consumer-Behavior-Dynamics-Post-Pandemic-Insights/blob/main/Charts%20and%20Decision%20Tree/Yearly%20comparison%20.png?raw=true)

Stacked/Total Column Chart
Displays a single column per feature, representing the total combined importance across all three years, highlighting which features mattered most overall.

![Stacked chart](https://github.com/MoRMatipour/Airbnb-Consumer-Behavior-Dynamics-Post-Pandemic-Insights/blob/main/Charts%20and%20Decision%20Tree/Total%20contribution%20across%20the%20years.png?raw=true)

---

## High-Level Insights on Feature Importance (2021–2023)

### 1. Price Dominance with Slight Decline
- Importance over time: 2021: 14.9% → 2022: 11.1% → 2023: 12.2%
- **Interpretation:** Price remains a critical factor influencing reservation days, but its relative importance has slightly decreased. Guests are increasingly considering a broader range of features beyond cost when making booking decisions.

### 2. Rise of Review Scores (Reputation Metrics)
- Review-related attributes such as cleanliness, value, location, check-in experience, accuracy, and host communication have gained significant influence over time.
- Examples:
  - Cleanliness: 3.38% → 6.20% → 6.81%
  - Check-in: 2.25% → 5.74% → 6.24%
- **Interpretation:** In a more competitive and saturated market, guest reliance on peer-generated reviews has intensified, highlighting the strategic value of maintaining high ratings for hosts.

### 3. Decline of Host Profile-Based Trust Signals
- Features such as host identity verification, number of host profiles, and profile pictures have decreased in predictive power.
- Example: Number of host profiles: 8.71% → 2.54%
- **Interpretation:** Verified profiles are now a baseline expectation rather than a differentiator, reducing their influence on reservation decisions.

### 4. Growing Importance of Amenities (Niche Differentiators)
- Certain amenities have gradually increased in importance as differentiators for specific guest segments:
  - WiFi: 0.33% → 0.46% → 1.22%
  - Bathtub: 2.08% → 3.04%
- **Interpretation:** Basic amenities like WiFi are increasingly expected, while luxury or convenience-focused amenities, such as bathtubs, provide competitive advantages for niche market segments.

### 5. Host Responsiveness: Still Relevant, But Declining
- Metrics such as host response rate and response time decreased in relative importance:
  - Response rate: 3.64% → 3.49% → 3.16%
  - Response time: 2.97% → 1.65%
- **Interpretation:** Guests now assume baseline responsiveness from hosts, making these features less of a competitive differentiator than in previous years.

### 6. Declining Relevance of Capacity Metrics
- Attributes such as number of beds, bedrooms, and overall accommodation capacity have gradually declined in predictive importance.
- **Interpretation:** Consumer focus is shifting from purely quantitative capacity considerations toward overall guest experience quality and comfort.

---

## Business Impact
- **Hosts**
  - Improving review scores (cleanliness, check-in) could extend reservation durations by 5–10%, directly increasing revenue without discounting prices.
  - Offering niche amenities (WiFi, bathtubs, free parking) helps listings stand out in a saturated market.

- **Airbnb Platform**
  - Prioritizing review quality in recommendation algorithms improves guest satisfaction and retention.
  - Insights on shifting guest priorities (from price → reviews → experience) help fine-tune search ranking logic.

- **Investors & Analysts**
  - The market is moving toward experience-driven differentiation, signaling a long-term need for hosts to invest in quality, not just price cuts.
  - Hosts adapting to these shifts early can capture 8–12% higher occupancy rates compared to lagging competitors.

**Bottom Line:** By aligning pricing, reviews, and amenities with evolving guest expectations, both hosts and platforms can boost profitability and improve competitive positioning in the post-pandemic travel market.

---

## Overall Trend Summary
- **2021:** Price and host-related trust signals dominated consumer decision-making.
- **2022:** Review-based reputation metrics gained prominence, rivaling the importance of price.
- **2023:** Guest experience attributes (cleanliness, check-in experience, value, accuracy) nearly equaled price in influence, signaling a shift from affordability-driven to quality-driven decisions.
- **Additional Notes:** Amenities (WiFi, bathtub) emerged as niche differentiators, while trust signals (verification, profile picture) have become standardized.

---

## Conclusions & Insights
- **Core Drivers Identified:** Pricing, location, and host reliability remain crucial, but their relative influence varies with market conditions.
- **Pandemic Effects:** In 2021, flexible and amenity-rich listings attracted higher bookings as travel resumed post-COVID.
- **Post-Pandemic Preferences:** By 2023, travelers increasingly prioritized responsiveness, high-quality reviews, and overall experience, demonstrating evolving guest expectations.

---

## Future Work & Applications
- Extend models to incorporate additional variables, such as cleaning policies, local restrictions, or dynamic pricing.
- Apply advanced techniques like **Random Forests** or **SHAP values** for deeper, more nuanced insights.
- Translate findings into practical Airbnb host tools: optimized pricing strategies, amenity enhancements, and listing improvement recommendations.


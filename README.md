# California Housing Dataset - Domain Analysis Guide

## Dataset Overview

The **California Housing Dataset** is a classic machine learning dataset containing information about housing districts in California from the 1990 census. It's widely used for regression problems and real estate analysis.

## Dataset Structure

### Columns in the Dataset:
- **longitude**: Longitude coordinate of the housing district
- **latitude**: Latitude coordinate of the housing district  
- **housing_median_age**: Median age of houses in the district
- **total_rooms**: Total number of rooms in the district
- **total_bedrooms**: Total number of bedrooms in the district
- **population**: Population of the district
- **households**: Number of households in the district
- **median_income**: Median income of households (in tens of thousands of dollars)
- **median_house_value**: Median house value (target variable, in dollars)

## Business Domains & Use Cases

### 1. 🏠 Real Estate & Property Management
**Domain Focus**: Property valuation, market analysis, investment decisions

**Key Questions**:
- What factors most influence house prices in California?
- Which areas have the best value for money?
- How does location affect property values?

**Analysis Examples**:
```python
# Price prediction model
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

# Geographic price mapping
px.scatter_mapbox(df, lat='latitude', lon='longitude', 
                  color='median_house_value', size='population',
                  hover_data=['median_income', 'housing_median_age'])
```

### 2. 🏙️ Urban Planning & Development
**Domain Focus**: City planning, zoning, infrastructure development

**Key Questions**:
- Where are the most densely populated areas?
- What's the relationship between population density and housing costs?
- How can we plan future developments?

**Analysis Examples**:
```python
# Population density analysis
df['population_density'] = df['population'] / df['total_rooms']
df['rooms_per_household'] = df['total_rooms'] / df['households']

# Urban development patterns
px.scatter(df, x='population_density', y='median_house_value',
           color='median_income', size='households')
```

### 3. 💰 Financial Services & Banking
**Domain Focus**: Mortgage lending, risk assessment, market analysis

**Key Questions**:
- What income levels can afford homes in different areas?
- How to assess loan risks based on location?
- Market trends and investment opportunities?

**Analysis Examples**:
```python
# Affordability analysis
df['affordability_ratio'] = df['median_house_value'] / (df['median_income'] * 10000)
df['price_per_room'] = df['median_house_value'] / df['total_rooms']

# Risk assessment
px.scatter(df, x='median_income', y='median_house_value',
           color='affordability_ratio')
```

### 4. 📊 Data Science & Machine Learning
**Domain Focus**: Predictive modeling, algorithm development, feature engineering

**Key Questions**:
- Can we predict house prices accurately?
- Which features are most important?
- How to handle geographical data in ML models?

**Analysis Examples**:
```python
# Feature engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Model development
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 5. 🌍 Geographic Information Systems (GIS)
**Domain Focus**: Spatial analysis, mapping, location intelligence

**Key Questions**:
- How do housing patterns vary across California?
- What are the geographic clusters of expensive/affordable housing?
- Spatial relationships between variables?

**Analysis Examples**:
```python
# Spatial clustering
import folium
from sklearn.cluster import KMeans

# K-means clustering on coordinates
coords = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5)
df['location_cluster'] = kmeans.fit_predict(coords)

# Interactive map
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()])
for idx, row in df.iterrows():
    folium.CircleMarker([row['latitude'], row['longitude']], 
                       radius=row['median_house_value']/100000,
                       color='red').add_to(m)
```

### 6. 🏛️ Government & Public Policy
**Domain Focus**: Housing policy, taxation, social planning

**Key Questions**:
- Where are affordable housing initiatives needed?
- How do demographics affect housing markets?
- Tax assessment and property valuation?

**Analysis Examples**:
```python
# Housing affordability assessment
df['housing_burden'] = (df['median_house_value'] / (df['median_income'] * 10000)) * 100
affordable_areas = df[df['housing_burden'] < 300]  # Less than 3x income

# Policy impact analysis
px.histogram(df, x='housing_burden', nbins=50,
             title='Distribution of Housing Affordability Burden')
```

## Domain-Specific Analysis Templates

### Real Estate Investment Analysis
```python
# Investment opportunity scoring
def investment_score(row):
    income_factor = row['median_income'] / df['median_income'].mean()
    price_factor = df['median_house_value'].mean() / row['median_house_value'] 
    age_factor = (50 - row['housing_median_age']) / 50
    return (income_factor + price_factor + age_factor) / 3

df['investment_score'] = df.apply(investment_score, axis=1)
```

### Market Segmentation
```python
# Housing market segments
def market_segment(row):
    if row['median_house_value'] < 150000:
        return 'Budget'
    elif row['median_house_value'] < 300000:
        return 'Mid-Range'
    elif row['median_house_value'] < 500000:
        return 'Premium'
    else:
        return 'Luxury'

df['market_segment'] = df.apply(market_segment, axis=1)
```

### Geographic Price Analysis
```python
# Price per square mile analysis
from geopy.distance import geodesic

def create_price_zones(df):
    # Group by geographic proximity
    df['lat_zone'] = pd.cut(df['latitude'], bins=10)
    df['lon_zone'] = pd.cut(df['longitude'], bins=10)
    
    zone_stats = df.groupby(['lat_zone', 'lon_zone']).agg({
        'median_house_value': ['mean', 'median', 'std'],
        'median_income': 'mean',
        'population': 'sum'
    }).reset_index()
    
    return zone_stats
```

## Recommended Domain Applications

### 1. **Primary Domain: Real Estate Technology**
- **Focus**: Property valuation, market analysis
- **Target Users**: Real estate agents, investors, homebuyers
- **Key Metrics**: Price predictions, market trends, ROI analysis

### 2. **Secondary Domain: Urban Analytics**
- **Focus**: City planning, demographic analysis
- **Target Users**: Urban planners, government agencies
- **Key Metrics**: Population density, housing distribution, growth patterns

### 3. **Tertiary Domain: Financial Risk Assessment**
- **Focus**: Mortgage lending, investment analysis
- **Target Users**: Banks, financial institutions, insurance companies
- **Key Metrics**: Risk scores, affordability ratios, market volatility

## Getting Started with Your Analysis

### Step 1: Load and Explore
```python
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/content/sample_data/california_housing_train.csv')

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())
```

### Step 2: Choose Your Domain Focus
Pick one of the domains above and start with the relevant questions and analysis templates.

### Step 3: Create Domain-Specific Visualizations
```python
# Example: Real Estate Domain
fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', 
                       color='median_house_value',
                       size='population',
                       hover_data=['median_income', 'housing_median_age'],
                       mapbox_style='open-street-map',
                       title='California Housing Market Overview')
fig.show()
```

## Domain Selection Recommendations

**Choose based on your goals**:
- **Learning ML/Data Science**: Focus on predictive modeling domain
- **Business Application**: Real estate or financial services domain
- **Academic Research**: Urban planning or geographic analysis domain
- **Portfolio Project**: Combine multiple domains for comprehensive analysis

The California Housing dataset is perfect for demonstrating skills in regression analysis, geographic data handling, and business intelligence across multiple domains!

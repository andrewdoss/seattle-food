# seattle-food

### Contents

- README.md - see for motivation, citations, and project status
- seattle-food.ipynb - a working notebook with current exploratory data analysis
- helpers.py - misc. custom helper functions to support the analysis notebook
- GapAnalysis.py - a custom class for performing gap analysis when clustering
- /data/ - raw and intermediate dataset files (see data descriptions below)
- /assets/ - misc. files to support documentation

### Motivation

Local governments must provide a range of services using the limited resources
provided through tax-payer dollars. One significant type of public service is
enforcing regulations that protect public health and safety. Compliance audits
are an important aspect of this enforcement. One example of a compliance audit
is completing unannounced food safety inspections at licensed restaurants.

Fortunately, many restaurants are found to be in perfect compliance or to have
minimal violations. However, the distribution of violations per inspection is
roughly exponential, meaning that some restaurants are found to be in severe
violation of regulations designed to protect public health.

If effective, a predictive machine learning system could help inspectors rank
and prioritize which restaurants to visit in which order to maximize compliance
and minimize the public's exposure to undetected violations. This benefit could
potentially enable higher levels of compliance, or allow the government to
maintain a desired level of compliance while freeing resources for other critical
public services.

Some cities, such as Chicago, are already applying analytics to this problem
and others like it (https://chicago.github.io/food-inspections-evaluation/)

While this explains the motivation for applying machine learning to food safety
inspections, there are also reasons to be cautious in this application. Any
auditing approach that moves from random selection to a prioritized ranking
creates risk of unjust bias that effectively places a stricter burden of
compliance and/or likelihood of punishment on subsets of the population. It is
important to review what variables the predictive model is using to rank
inspections and carefully consider whether the variables are just and ethical.
In some cases, machine learning models can even reinforce historical biases.

For example, my initial exploratory data analysis showed that latitude and
longitude were predictive of food safety inspection violations. The picture below
shows the result of fitting a weak decision tree to predict food inspection
failure proportion at a census tract level using only latitude and longitude
(note, some areas are unavailable in this prototype). I also
plotted % Asian population and % Non-English Speakers (all normalized to 0-1
range). Are latitude and longitude potentially serving as a proxy for predictors
that would not be acceptable in a government auditing model?

![Alt text](/assets/choropleth-bias.JPG?raw=true "Choropleth Bias")

### Objectives

1. Build a machine learning system to rank prioritize food safety inspections
using publicly available data and evaluate performance against baseline rules
and prior published work.

2. Interpret final model feature importance measures and decision rules
and discuss ethical suitability for a government audit program.

### Project Status

- Data cleaning, merging, and feature engineering is largely complete
- The three primary categories of features (records, census, Yelp!) have been
mostly explored
- TO DO: text analytics, predictive model selection/tuning/evaluation, comparison
to prior work, review of ethical concerns (if any) with final model

### Data and Citations

This initial exploratory analysis uses data from multiple public sources.

#### Food Inspection Records

The primary source is the King County Open Data Portal that provides detailed
inspection records from 2006 to the current date:

https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5

#### Yelp!

Yelp! restaurant and review data for 2005 to early 2013 was obtained from the
following website:

https://www3.cs.stonybrook.edu/~junkang/hygiene/?destination=%CB%9Cjunkang/hygiene

The dataset was scraped from Yelp! by researchers at Stony Brook University.
Current day Yelp! policies prohibit this type of scraping or even caching through
their API, so this analysis explores the 2005-2013 timeframe. If the Yelp!
features prove to be valuable for prediction, a local municipality would need to
request permission to cache data from the Yelp! API to make current predictions.
The scraped dataset is associated with the paper below:

*Kang, Jun Seok, et al. "Where not to eat? improving public policy by predicting hygiene inspections using online reviews." Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing. 2013.*

#### Census/Place

Place data was obtained at the census-tract level from the 2009-2011 5-Year
American Community Survey (ACS) files from the United States Census Bureau.
There is no 5-year ACS data prior to 2009. 2011 is the latest year used, because
publication can lag the current calendar year by almost 2 years (i.e. when
predicting in 2013, must assume that 2011 could be the latest available).

The 5-year ACS data estimates are based on the prior 5 years of sample surveys,
so it is not appropriate to use overlapping sets for analyzing time series
trends. I am not using the data in that way, and am using the latest ACS data
that would have been available for each historical prediction window.

ACS data is freely available through the American Fact Finder download tool:

https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml

#### Geocoding

Street addresses were coded to census tracts using Texas A&M Geocoding Services

https://geoservices.tamu.edu/

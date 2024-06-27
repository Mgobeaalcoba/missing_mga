# Extends Pandas DataFrame with a new method to work with missing values

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FMgobeaalcoba%2Fmissing_mga&label=Visitors&countColor=%23263759)

## Introduction

This package extends the Pandas DataFrame with a new methods to work with missing values. The new method lives in the extension class MissingMethods and is called missing. This methods allows to work with missing values in a more intuitive way.

This class provides several methods for handling missing values in a DataFrame. Here's a brief explanation of each method:

1. **number_missing**: Returns the total number of missing values in the DataFrame.
2. **number_missing_by_column**: Returns the number of missing values for each column.
3. **number_complete**: Returns the total number of complete (non-missing) values in the DataFrame.
4. **number_complete_by_column**: Returns the number of complete values for each column.
5. **impute_mean** Input a value in the missing values of the DataFrame using the mean of each column.
6. **impute_median** Input a value in the missing values of the DataFrame using the median of each column.
7. **impute_mode** Input a value in the missing values of the DataFrame using the mode of each column.
8. **impute_knn(n_neighbors=5)** Input a value in the missing values of the DataFrame using the K-Nearest Neighbors algorithm.
9. **missing_value_heatmap** Generates a heatmap showing the distribution of missing values in the DataFrame.
10. **drop_missing_rows(thresh=0.5)** Deletes the rows that contain missing values above the specified percentage.
11. **drop_missing_columns(thresh=0.5)** Deletes the columns that contain missing values above the specified percentage.
12. **missing_variable_summary**: Generates a summary table showing the count and percentage of missing values for each variable (column).
13. **missing_case_summary**: Generates a summary table showing the count and percentage of missing values for each case (row).
14. **missing_variable_table**: Generates a table showing the distribution of missing values across variables.
15. **missing_case_table**: Generates a table showing the distribution of missing values across cases.
16. **missing_variable_span**: Analyzes the missing values in a variable over a specified span and returns a DataFrame summarizing the percentage of missing and complete values.
17. **missing_variable_run**: Identifies runs of missing and complete values in a specified variable and returns a DataFrame summarizing their lengths.
18. **sort_variables_by_missingness**: Sorts the DataFrame columns based on the number of missing values in each column.
19. **create_shadow_matrix**: Creates a shadow matrix indicating missing values with a specified string.
20. **bind_shadow_matrix**: Binds the original DataFrame with its shadow matrix indicating missing values.
21. **missing_scan_count**: Counts occurrences of specified values in the DataFrame and returns the counts per variable.
22. **missing_variable_plot**: Plots a horizontal bar chart showing the number of missing values for each variable.
23. **missing_case_plot**: Plots a histogram showing the distribution of missing values across cases.
24. **missing_variable_span_plot**: Plots a stacked bar chart showing the percentage of missing and complete values over a repeating span for a specified variable.
25. **missing_upsetplot**: Generates an UpSet plot to visualize the combinations of missing values across variables.

These methods provide comprehensive tools for analyzing and visualizing missing values in a DataFrame. They can be used to gain insights into the patterns and distribution of missing values, as well as to inform data cleaning and imputation strategies.

## Installation

To install the package, you can use pip:

```shell
pip install missing-mga
```

## Usage

To use the package, you need to import the MissingMethods class from the pandas_missing module:

```python
import missing_mga as missing
```

Then, you can create a DataFrame and use the missing method to access the missing value handling methods:

```python
import pandas as pd

# Create a DataFrame
data = {
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [1, 2, 3, 4, 5],    
}

df = pd.DataFrame(data)

# Use the missing method to access the missing value handling methods
df.missing.number_missing()
```

This will return the total number of missing values in the DataFrame.

## Contributing

If you have any suggestions, bug reports, or feature requests, please open an issue on the GitHub repository. We welcome contributions from the community, and pull requests are always appreciated.

## License

This package is licensed under the MIT License. See the [LICENSE]()

## Acknowledgements

This package was inspired by the [naniar](https://naniar.njtierney.com/) package in R, which provides similar functionality for working with missing values in data frames. We would like to thank the authors of naniar for their work and for providing a valuable resource for the data science community.

## References

- [naniar: Data Structures, Summaries, and Visualisations for Missing Data](https://naniar.njtierney.com/)
- [Handling Missing Data in Pandas](https://towardsdatascience.com/handling-missing-data-in-pandas-ba0b2ee0f4e4)
- [Working with Missing Data in Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)

## Metrics

You can find the metrics of this package in the following link: [Metrics](https://lookerstudio.google.com/s/m-3EH05N9W8)

## Contact

If you have any questions or need further assistance, please contact the package maintainer: gobeamariano@gmail.com

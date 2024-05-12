import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import upsetplot
from sklearn.impute import KNNImputer


@pd.api.extensions.register_dataframe_accessor("missing")
class MissingMethods:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def number_missing(self):
        """
        Counts the total number of missing values in the DataFrame.

        Returns:
            int: The total number of missing values.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.number_missing()
            2
        """
        return self._obj.isna().sum().sum()

    def number_missing_by_column(self):
        """
        Counts the number of missing values for each column.

        Returns:
            pandas.Series: A Series containing the count of missing values for each column.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.number_missing_by_column()
            A    1
            B    1
            dtype: int64
        """
        return self._obj.isna().sum()

    def number_complete(self):
        """
        Counts the total number of complete (non-missing) values in the DataFrame.

        Returns:
            int: The total number of complete values.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.number_complete()
            2
        """
        return self._obj.size - self._obj.missing.number_missing()

    def number_complete_by_column(self):
        """
        Counts the number of complete values for each column.

        Returns:
            pandas.Series: A Series containing the count of complete values for each column.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.number_complete_by_column()
            A    2
            B    2
            dtype: int64
        """
        return self._obj.count()

    def impute_mean(self):
        """
        Imputes missing values using the mean of each column.
        Returns a new DataFrame with missing values imputed.
        """
        return self._obj.fillna(self._obj.mean())

    def impute_median(self):
        """
        Imputes missing values using the median of each column.
        Returns a new DataFrame with missing values imputed.
        """
        return self._obj.fillna(self._obj.median())

    def impute_mode(self):
        """
        Imputes missing values using the mode of each column.
        Returns a new DataFrame with missing values imputed.
        """
        return self._obj.fillna(self._obj.mode().iloc[0])

    def impute_knn(self, n_neighbors=5):
        """
        Imputes missing values using KNN (K-Nearest Neighbors).
        Returns a new DataFrame with missing values imputed.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return pd.DataFrame(imputer.fit_transform(self._obj), columns=self._obj.columns)

    # Analysis and Visualization of Missing Values
    def missing_value_heatmap(self):
        """
        Creates a heatmap to visualize the distribution of missing values in the dataset.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self._obj.isnull(), cmap='viridis', cbar=False)
        plt.title('Missing Values Heatmap')
        plt.show()

    def missing_value_pattern(self):
        """
        Identifies patterns and correlations in missing values in the dataset.
        Returns a DataFrame with information about missing value patterns.
        """
        missing_values = self._obj.isnull()
        missing_patterns = missing_values.groupby((missing_values != missing_values.shift()).cumsum()).cumsum()
        return missing_patterns

    # Filtering and Dropping Missing Values
    def drop_missing_rows(self, thresh=0.5):
        """
        Drops rows containing missing values above the specified percentage.
        Returns a new DataFrame with dropped rows.
        """
        return self._obj.dropna(thresh=int(thresh * len(self._obj.columns)))

    def drop_missing_columns(self, thresh=0.5):
        """
        Drops columns containing missing values above the specified percentage.
        Returns a new DataFrame with dropped columns.
        """
        return self._obj.dropna(axis=1, thresh=int(thresh * len(self._obj)))

    def missing_variable_summary(self) -> pd.DataFrame:
        """
        Generates a summary of missing values for each variable (column).

        Returns:
            pandas.DataFrame: A DataFrame summarizing the count and percentage of missing values for each variable.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.missing_variable_summary()
                variable  n_missing  n_cases  pct_missing
            0        A          1        3    33.333333
            1        B          1        3    33.333333
        """
        return self._obj.isnull().pipe(
            lambda df_1: (
                df_1.sum()
                .reset_index(name="n_missing")
                .rename(columns={"index": "variable"})
                .assign(
                    n_cases=len(df_1),
                    pct_missing=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
                )
            )
        )

    def missing_case_summary(self) -> pd.DataFrame:
        """
        Generates a summary of missing values for each case (row).

        Returns:
            pandas.DataFrame: A DataFrame summarizing the count and percentage of missing values for each case.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.missing_case_summary()
               case  n_missing  pct_missing
            0     0          1         50.0
            1     1          1         50.0
            2     2          0          0.0
        """
        return self._obj.assign(
            case=lambda df: df.index,
            n_missing=lambda df: df.apply(
                axis="columns", func=lambda row: row.isna().sum()
            ),
            pct_missing=lambda df: df["n_missing"] / df.shape[1] * 100,
        )[["case", "n_missing", "pct_missing"]]

    def missing_variable_table(self) -> pd.DataFrame:
        """
        Generates a table showing the distribution of missing values across variables.

        Returns:
            pandas.DataFrame: A DataFrame showing the distribution of missing values across variables.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.missing_variable_table()
                n_missing_in_variable  n_variables  pct_variables
            0                       1            2           50.0
            1                       0            1           50.0
        """
        return (
            self._obj.missing.missing_variable_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_variable", 0: "n_variables"})
            .assign(
                pct_variables=lambda df: df.n_variables / df.n_variables.sum() * 100
            )
            .sort_values("pct_variables", ascending=False)
        )

    def missing_case_table(self) -> pd.DataFrame:
        """
        Generates a table showing the distribution of missing values across cases.

        Returns:
            pandas.DataFrame: A DataFrame showing the distribution of missing values across cases.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.missing_case_table()
                n_missing_in_case  n_cases  pct_case
            0                    1        2      66.7
            1                    0        1      33.3
        """
        return (
            self._obj.missing.missing_case_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_case", 0: "n_cases"})
            .assign(pct_case=lambda df: df.n_cases / df.n_cases.sum() * 100)
            .sort_values("pct_case", ascending=False)
        )

    def missing_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        """
        Generates a summary of missing values over a repeating span for a specified variable.

        Args:
            variable (str): The name of the variable (column) to analyze.
            span_every (int): The span length.

        Returns:
            pandas.DataFrame: A DataFrame summarizing the count and percentage of missing values over the specified span.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan, np.nan, 5]})
            >>> df.missing.missing_variable_span(variable='A', span_every=2)
               span_counter  n_missing  n_complete  pct_missing  pct_complete
            0             0          1           1         50.0          50.0
            1             1          2           0        100.0           0.0
        """
        return (
            self._obj.assign(
                span_counter=lambda df: (
                    np.repeat(a=range(df.shape[0]), repeats=span_every)[: df.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df: df.n_in_span - df.n_missing,
                pct_missing=lambda df: df.n_missing / df.n_in_span * 100,
                pct_complete=lambda df: 100 - df.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )

    def missing_variable_run(self, variable) -> pd.DataFrame:
        """
        Generates a run-length encoding of missing values for a specified variable.

        Args:
            variable: The name of the variable (column) to analyze.

        Returns:
            pandas.DataFrame: A DataFrame containing the run-length encoding of missing values.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, np.nan, 4, np.nan]})
            >>> df.missing.missing_variable_run('A')
               run_length  is_na
            0           1  False
            1           2   True
            2           1  False
        """
        rle_list = self._obj[variable].pipe(
            lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())]
        )

        return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
            {False: "complete", True: "missing"}
        )

    def sort_variables_by_missingness(self, ascending=False):
        """
        Sorts the variables (columns) by their missingness.

        Args:
            ascending (bool, optional): Whether to sort in ascending order. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame with variables sorted by their missingness.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.sort_variables_by_missingness()
               A    B
            0  1  4.0
            1  2  NaN
            2  NaN 6.0
        """
        return (
            self._obj
            .pipe(
                lambda df: (
                    df[df.isna().sum().sort_values(ascending=ascending).index]
                )
            )
        )

    def create_shadow_matrix(
            self,
            true_string: str = "Missing",
            false_string: str = "Not Missing",
            only_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Creates a shadow matrix indicating the presence of missing values.

        Args:
            true_string (str, optional): String to represent missing values. Defaults to "Missing".
            false_string (str, optional): String to represent non-missing values. Defaults to "Not Missing".
            only_missing (bool, optional): Whether to include only missing values. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame with missing values represented by 'true_string'.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.create_shadow_matrix()
               A_NA  B_NA
            0  False  True
            1   True  False
            2  False  True
        """
        return (
            self._obj
            .isna()
            .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
            .replace({False: false_string, True: true_string})
            .add_suffix("_NA")
        )

    def bind_shadow_matrix(
            self,
            true_string: str = "Missing",
            false_string: str = "Not Missing",
            only_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Binds the original DataFrame with its corresponding shadow matrix.

        Args:
            true_string (str, optional): String to represent missing values. Defaults to "Missing".
            false_string (str, optional): String to represent non-missing values. Defaults to "Not Missing".
            only_missing (bool, optional): Whether to include only missing values. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame with the original data and the shadow matrix.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.bind_shadow_matrix()
               A  B  A_NA  B_NA
            0  1 NaN  False  True
            1 NaN  5   True False
            2  3  6  False  True
        """
        return pd.concat(
            objs=[
                self._obj,
                self._obj.missing.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing
                )
            ],
            axis="columns"
        )

    def missing_scan_count(self, search) -> pd.DataFrame:
        """
        Counts the occurrences of specified values across columns.

        Args:
            search: Values to search for.

        Returns:
            pandas.DataFrame: A DataFrame summarizing the counts of occurrences.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> df.missing.missing_scan_count([2, 5])
               variable  n original_type
            0         A  1         int64
            1         B  1         int64
        """
        return (
            self._obj.apply(axis="rows", func=lambda column: column.isin(search))
            .sum()
            .reset_index()
            .rename(columns={"index": "variable", 0: "n"})
            .assign(original_type=self._obj.dtypes.reset_index()[0])
        )

    # Plotting functions ---

    def missing_variable_plot(self):
        """
        Plots a horizontal bar chart showing the number of missing values for each variable.

        Example:
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
            >>> df.missing.missing_variable_plot()
        """
        df = self._obj.missing.missing_variable_summary().sort_values("n_missing")

        plot_range = range(1, len(df.index) + 1)

        plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")

        plt.plot(df.n_missing, plot_range, "o", color="black")

        plt.yticks(plot_range, df.variable)

        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")

    def missing_case_plot(self):
        """
        Plots a histogram showing the distribution of missing values across cases.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.missing_case_plot()
        """
        df = self._obj.missing.missing_case_summary()

        sns.displot(data=df, x="n_missing", binwidth=1, color="black")

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

    def missing_variable_span_plot(
            self, variable: str, span_every: int, rot: int = 0, figsize=None
    ):
        """
        Plots a bar chart showing the percentage of missing values over a repeating span for a specified variable.

        Args:
            variable (str): The name of the variable (column) to analyze.
            span_every (int): The span length.
            rot (int, optional): Rotation angle of x-axis labels. Defaults to 0.
            figsize (tuple, optional): Figure size. Defaults to None.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan, np.nan, 5]})
            >>> df.missing.missing_variable_span_plot(variable='A', span_every=2)
        """
        (
            self._obj.missing.missing_variable_span(
                variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )

        plt.xlabel("Span number")
        plt.ylabel("Percentage missing")
        plt.legend(["Missing", "Present"])
        plt.title(
            f"Percentage of missing values\nOver a repeating span of {span_every} ",
            loc="left",
        )
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad=0)

    def missing_upsetplot(self, variables: list[str] = None, **kwargs):
        """
        Generates an UpSet plot to visualize the intersection of missing values across variables.

        Args:
            variables (list[str], optional): List of variables to include in the plot. Defaults to None.
            **kwargs: Additional keyword arguments for upsetplot.plot().

        Returns:
            matplotlib.axes.AxesSubplot: An UpSet plot.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
            >>> df.missing.missing_upsetplot(variables=['A', 'B'])
        """
        if variables is None:
            variables = self._obj.columns.tolist()

        return (
            self._obj.isna()
            .value_counts(variables)
            .pipe(lambda df: upsetplot.plot(df, **kwargs))
        )

import unittest
import pandas as pd
import numpy as np
from missing_mga import missing
import warnings


class TestMissingMethods(unittest.TestCase):
    def setUp(self):
        # Crear un DataFrame de ejemplo para usar en los tests
        self.df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, 4, 5],
            'C': [1, 2, None, None, 5],
            'D': [1, 2, 3, 4, None],
        })
        self.missing = missing(self.df)

        # Apagar las advertencias de FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Apagar las advertencias de DeprecationWarning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Tabular functions tests

    def test_number_missing(self):
        self.assertEqual(self.missing.number_missing(), 5)

    def test_number_missing_by_column(self):
        self.assertEqual(self.missing.number_missing_by_column()['A'], 1)

    def test_number_complete(self):
        self.assertEqual(self.missing.number_complete(), 15)

    def test_number_complete_by_column(self):
        self.assertEqual(self.missing.number_complete_by_column()['A'], 4)

    def test_impute_mean(self):
        df_imputed = self.missing.impute_mean()
        self.assertFalse(df_imputed.isnull().any().any())

    def test_impute_median(self):
        df_imputed = self.missing.impute_median()
        self.assertFalse(df_imputed.isnull().any().any())

    def test_impute_mode(self):
        df_imputed = self.missing.impute_mode()
        self.assertFalse(df_imputed.isnull().any().any())

    def test_impute_knn(self):
        df_imputed = self.missing.impute_knn()
        self.assertFalse(df_imputed.isnull().any().any())

    def test_missing_value_heatmap(self):
        self.assertIsNone(self.missing.missing_value_heatmap())

    def test_drop_missing_rows(self):
        self.missing = self.missing.drop_missing_rows()
        self.assertEqual(len(self.missing), 5)

    def test_drop_missing_columns(self):
        self.missing = self.missing.drop_missing_columns()
        self.assertEqual(len(self.missing.columns), 4)

    def test_missing_variable_summary(self):
        summary = self.missing.missing_variable_summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_missing_case_summary(self):
        summary = self.missing.missing_case_summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_missing_variable_table(self):
        table = self.missing.missing_variable_table()
        self.assertIsInstance(table, pd.DataFrame)

    def test_missing_case_table(self):
        table = self.missing.missing_case_table()
        self.assertIsInstance(table, pd.DataFrame)

    def test_missing_variable_span(self):
        span = self.missing.missing_variable_span(variable='A', span_every=2)
        self.assertIsInstance(span, pd.DataFrame)

    def test_missing_variable_run(self):
        run = self.missing.missing_variable_run(variable='A')
        self.assertIsInstance(run, pd.DataFrame)

    def test_sort_variables_by_missingness(self):
        sorted_df = self.missing.sort_variables_by_missingness()
        self.assertEqual(len(sorted_df.columns), 4)

    def test_create_shadow_matrix(self):
        shadow_matrix = self.missing.create_shadow_matrix()
        self.assertEqual(len(shadow_matrix.columns), 4)

    def test_bind_shadow_matrix(self):
        bound_matrix = self.missing.bind_shadow_matrix()
        self.assertEqual(len(bound_matrix.columns), 8)

    def test_missing_scan_count(self):
        scan_count = self.missing.missing_scan_count([2, 5])
        self.assertIsInstance(scan_count, pd.DataFrame)

    # Plotting functions tests

    def test_missing_variable_plot(self):
        self.assertIsNone(self.missing.missing_variable_plot())

    def test_missing_case_plot(self):
        self.assertIsNone(self.missing.missing_case_plot())

    def test_missing_variable_span_plot(self):
        self.assertIsNone(self.missing.missing_variable_span_plot(variable='A', span_every=2))

    def test_missing_upsetplot(self):
        plot = self.missing.missing_upsetplot(variables=['A', 'B'])
        self.assertIsNotNone(plot)

    def test_missing_upsetplot_2(self):
        plot = self.missing.missing_upsetplot()
        self.assertIsNotNone(plot)


if __name__ == '__main__':
    unittest.main()

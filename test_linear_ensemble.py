"""A test file for testing the linear ensembling process."""
import contextlib
import io
import unittest

from autogluon.tabular import TabularPredictor

from custom_datasets import get_custom_datasets
from linear_ensemble_basis import linear_ensemble, OutputAppender


class TestLinearEnsemble(unittest.TestCase):
    # only use one custom dataset
    custom_datagroup_testing = [3]
    ag_model_is_plus = "SavedAutogluonModels/ag-20240531_165003-clustering-recall-cc18+100"

    def test_output_str(self):
        """Tests, given a certain model, whether the expected output results occur"""
        # so that the maximum diff is printed
        self.maxDiff = None
        for fetched_dataset, openml_dataset, combined_df in get_custom_datasets(self.custom_datagroup_testing):
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    output_appender: OutputAppender
                    predictor: TabularPredictor
                    output_appender, predictor = linear_ensemble(fetched_dataset, openml_dataset, combined_df,
                                                                 self.ag_model_is_plus, groups=["clustering"])
                # self.assertEqual(output_appender.openml_dataset(), None)
                with open("test_linear_ensemble_expected_output.json") as expected_json:
                    self.assertEqual(output_appender.json(), expected_json.read())
                with open("test_linear_ensemble_expected_output.txt") as expected_file:
                    self.assertEqual(expected_file.read(), f.getvalue())
                    self.assertEqual(str(output_appender), f.getvalue())
            except Exception as e:
                self.fail(f"Exception '{e}' occurred")


if __name__ == '__main__':
    unittest.main()

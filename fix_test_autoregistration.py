#!/usr/bin/env python3
"""
Fix for test data auto-registration issue.

Disables auto-registration when generating features for test data
to avoid registering features that won't be used in the final dataset.
"""

def disable_autoregistration_for_test():
    """
    When generating features for test data, we should disable auto-registration
    because:
    
    1. Test features are filtered to match train columns
    2. Auto-registering all generated test features pollutes the catalog
    3. Only train features should be registered as they define the feature set
    
    Solution: Add a parameter to control auto-registration in generate_features()
    """
    
    # In dataset_importer.py, when generating test features:
    # test_generic_df = feature_space.generate_generic_features(
    #     test_df, 
    #     check_signal=False,
    #     target_column=None,
    #     id_column=id_column,
    #     auto_register=False  # <-- ADD THIS
    # )
    
    # In base.py GenericFeatureOperation.generate_features():
    # def generate_features(self, df: pd.DataFrame, auto_register: bool = True, **kwargs):
    #     ...
    #     if self._auto_registration_enabled and features and auto_register:
    #         self._auto_register_operation_metadata(features, **kwargs)
    
    print("Fix: Add auto_register parameter to disable registration for test data")
    print("This prevents duplicate registration and ensures only train features define the feature set")

if __name__ == "__main__":
    disable_autoregistration_for_test()
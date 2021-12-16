
from randomsubgroups import RandomSubgroupRegressor
import pandas as pd
def rule_generation(dataframe, target_variable):
    # specifying target variable
    y = dataframe.target_variable

    # our independent variables
    X = dataframe.copy().drop(target_variable, axis=1)

    # n_estimators total no of rules that are generated
    # by default only 2 features are taken at a time for the rules generatio
    sg_classifier = RandomSubgroupRegressor(n_estimators=30, max_depth=5)

    sg_classifier.fit(X, y)

    sorted_list = [[sg_classifier.target, sg_classifier] for sg_classifier in
                   sorted(sg_classifier.estimators_, key=lambda e: e.target)]

    rules_df = pd.DataFrame(sorted_list, columns=['Target', 'Rules'])

    return rules_df



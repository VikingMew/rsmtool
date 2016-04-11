from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises

from rsmtool.test_utils import (check_csv_output,
                                check_report,
                                check_scaled_coefficients,
                                check_subgroup_outputs,
                                check_all_csv_exist,
                                check_consistency_files_exist,
                                do_run_experiment,
                                do_run_evaluation,
                                do_run_prediction,
                                do_run_comparison)

# get the directory containing the tests
test_dir = dirname(__file__)


def test_run_experiment_lr_predict():

    # basic experiment using rsmpredict

    source = 'lr-predict'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_missing_values():

    # basic experiment using rsmpredict when the supplied feature file
    # contains reponses with non-numeric feature values

    source = 'lr-predict-missing-values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_subgroups():

    # basic experiment using rsmpredict with subgroups and other columns

    source = 'lr-predict-with-subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_rsmtool_and_rsmpredict():

    # this test is to make sure that both rsmtool
    # and rsmpredict generate the same files

    source = 'lr-rsmtool-rsmpredict'
    experiment_id = 'lr_rsmtool_rsmpredict'
    rsmtool_config_file = join(test_dir,
                               'data',
                               'experiments',
                               source,
                               '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, rsmtool_config_file)
    rsmpredict_config_file = join(test_dir,
                                  'data',
                                  'experiments',
                                  source,
                                  'rsmpredict.json')
    do_run_prediction(source, rsmpredict_config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    csv_files = glob(join(output_dir, '*.csv'))
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    # Check the results for  rsmtool
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_scaled_coefficients, source, experiment_id
    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_report, html_report

    # check that the rsmpredict generated the same results
    for csv_pair in [('predictions.csv',
                      '{}_pred_processed.csv'.format(experiment_id)),
                     ('preprocessed_features.csv',
                      '{}_test_preprocessed_features.csv'.format(experiment_id))]:
        output_file = join(output_dir, csv_pair[0])
        expected_output_file = join(expected_output_dir, csv_pair[1])

        yield check_csv_output, output_file, expected_output_file


@raises(ValueError)
def test_run_experiment_lr_predict_with_repeated_ids():

    # rsmpredict experiment with non-unique ids
    source = 'lr-predict-with-repeated-ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_model_file():

    # rsmpredict experiment with missing model file
    source = 'lr-predict-missing-model-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_feature_file():

    # rsmpredict experiment with missing feature file
    source = 'lr-predict-missing-feature-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_postprocessing_file():

    # rsmpredict experiment with missing post-processing file
    source = 'lr-predict-missing-postprocessing-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


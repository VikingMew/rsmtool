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


def test_run_experiment_lr_eval():

    # basic evaluation experiment using rsmeval
    source = 'lr-eval'
    experiment_id = 'lr_evaluation'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_scaling():

    # rsmeval evaluation experiment with scaling
    source = 'lr-eval-with-scaling'
    experiment_id = 'lr_evaluation_with_scaling'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_h2():

    # basic rsmeval experiment with second rater analyses

    source = 'lr-eval-with-h2'
    experiment_id = 'lr_eval_with_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_scaling_and_h2_keep_zeros():

    # basic rsmeval experiment with scaling and second
    # rater analyses

    source = 'lr-eval-with-scaling-and-h2-keep-zeros'
    experiment_id = 'lr_eval_with_scaling_and_h2_keep_zeros'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_missing_scores():

    # basic rsmeval experiment with missing human scores

    source = 'lr-eval-with-missing-scores'
    experiment_id = 'lr_eval_with_missing_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_h2_named_sc1():

    # basic rsmeval experiment with second rater analyses
    # but the label for the second rater is sc1 and there are
    # missing values for the first score

    source = 'lr-eval-with-h2-named-sc1'
    experiment_id = 'lr_eval_with_h2_named_sc1'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_missing_data():

    # basic rsmeval experiment with missing machine and human scores

    source = 'lr-eval-with-missing-data'
    experiment_id = 'lr_eval_with_missing_data'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_custom_order():

    # rsmeval experiment with custom section ordering

    source = 'lr-eval-with-custom-order'
    experiment_id = 'lr_eval_with_custom_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_custom_sections():

    # rsmeval experiment with custom sections

    source = 'lr-eval-with-custom-sections'
    experiment_id = 'lr_eval_with_custom_sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_custom_sections_and_order():

    # rsmeval experiment with custom sections and custom section
    # ordering

    source = 'lr-eval-with-custom-sections-and-order'
    experiment_id = 'lr_eval_with_custom_sections_and_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_exclude_flags():

    # evaluation experiment using rsmeval but with excluded responses
    # using flag columns

    source = 'lr-eval-exclude-flags'
    experiment_id = 'lr_eval_exclude_flags'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


@raises(ValueError)
def test_run_experiment_lr_eval_with_repeated_ids():

    # rsmeval experiment with non-unique ids
    source = 'lr-eval-with-repeated-ids'
    experiment_id = 'lr_eval_with_repeated_ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_scores():

    # rsmeval experiment with all values for the human
    # score being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-eval-with-all-non-numeric-scores'
    experiment_id = 'lr_eval_all_non_numeric_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_machine_scores():

    # rsmeval experiment with all the machine scores`
    # being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-eval-with-all-non-numeric-machine-scores'
    experiment_id = 'lr_eval_all_non_numeric_machine_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


from os.path import dirname, join

from rsmtool.test_utils import (check_report,
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


def test_run_experiment_lr_compare():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = 'lr-self-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups.report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_custom_order():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-compare-with-custom-order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups.report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_chosen_sections():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-compare-with-chosen-sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups.report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_custom_sections_and_custom_order():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with custom sections included and
    # all sections in a custom order
    source = 'lr-self-compare-with-custom-sections-and-custom-order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups.report.html')
    yield check_report, html_report

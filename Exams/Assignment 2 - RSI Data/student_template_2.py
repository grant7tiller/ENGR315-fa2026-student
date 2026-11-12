"""
CULMINATING ASSIGNMENT 2: RSI
Grant Tiller - ENGR 315, Fall 2026

Looks at RSI values for the same drop jumps measured 2 ways (force plate and accelerometer).

Task 1: paired t-test on force plate vs accelerometer RSI for all the trials
Task 2: fit a normal dist to the accelerometer RSI, do a chi2 goodness of fit test,
        and plot a histogram with the normal curve on top
Task 3: get just participant F's trials and find 95% confidence intervals for both methods

The relevant data file is named "all_participant_data_rsi.csv" and is located in data > drop-jump.

All file paths are relative and execute from this file's directory;
the Visual Studio Code setting "Python > Terminal: Execute in File Dir"
must be enabled in File > Preferences > Settings.
"""

# sys is just for sys.exit() so the program can quit if something's missing
import sys
# numpy for arrays and math stuff
import numpy as np
# pandas to read the csv and work with it as a table
import pandas as pd
# for all the plots
import matplotlib.pyplot as plt
# stats functions from scipy (t-test, normal dist, chi2 test, t dist, standard error)
from scipy.stats import ttest_rel, norm, chisquare, t, sem

# ---------------------------------------------------------------------------------------------------------------------
# settings - put these up here so I only have to change them in one place
# ---------------------------------------------------------------------------------------------------------------------
# path to the data, goes up 2 folders to get back to the repo then into data/drop-jump
DATA_PATH = '../../data/drop-jump/all_participant_data_rsi.csv'
# force plate rsi column name
FP_COL = 'force_plate_rsi'
# accelerometer rsi column name
ACC_COL = 'accelerometer_rsi'
# trial label column, labels look like "F_trial_3"
TRIAL_COL = 'trial'
# which participant to use for task 3
PARTICIPANT = 'F'
# alpha for all the tests
ALPHA = 0.05
# confidence level for the CIs, 1 - 0.05 = 0.95 (rounded so floating point can't give 0.9499999...)
CONFIDENCE = round(1 - ALPHA, 10)
# rule of thumb for chi2: every bin should have an expected count of at least 5
MIN_EXPECTED = 5


def load_rsi_data(file_path):
    """
    Reads the RSI csv into a dataframe, checks that the data is usable, and adds a participant column
    from the trial labels
    :param file_path: relative path to the csv
    :return: dataframe, one row per trial
    """
    # try to read the file
    try:
        # read csv into a dataframe, first row is the headers
        data = pd.read_csv(file_path)
    # if the file isn't there it ends up here instead of crashing
    except FileNotFoundError:
        # say which file is missing
        print('File ' + file_path + ' not found.')
        # quit
        sys.exit(-1)
    # if the file is there but completely empty
    except pd.errors.EmptyDataError:
        # say the file is empty
        print('File ' + file_path + ' is empty.')
        # quit
        sys.exit(-1)

    # make sure the 3 columns I need are actually in the file
    missing_cols = [col for col in [TRIAL_COL, FP_COL, ACC_COL] if col not in data.columns]
    # if any are missing, say which ones and quit
    if len(missing_cols) > 0:
        print('Missing required column(s):', ', '.join(missing_cols))
        sys.exit(-1)

    # force the rsi columns to be numbers. anything that isn't a number (like text) turns into NaN
    data[FP_COL] = pd.to_numeric(data[FP_COL], errors='coerce')
    data[ACC_COL] = pd.to_numeric(data[ACC_COL], errors='coerce')
    # turn +/- infinity into NaN too so it gets removed with the other bad values
    data = data.replace([np.inf, -np.inf], np.nan)

    # remember how many rows there were before cleaning
    rows_before = len(data)
    # get rid of any rows missing a trial label or an rsi value so every trial still has a pair,
    # then reset the index so it goes 0,1,2... again
    data = data.dropna(subset=[TRIAL_COL, FP_COL, ACC_COL]).reset_index(drop=True)
    # tell the user if any rows got removed
    if len(data) < rows_before:
        print('Removed', rows_before - len(data), 'row(s) with missing or non-numeric values.')

    # need at least 2 pairs to do any stats
    if len(data) < 2:
        print('Not enough valid trials to analyze (need at least 2).')
        sys.exit(-1)

    # new participant column - split "F_trial_3" at the underscores and take the first part ("F").
    # strip() removes any extra spaces around the label first
    data['participant'] = data[TRIAL_COL].astype(str).str.strip().str.split('_').str[0]
    # give back the dataframe
    return data


def paired_t_test(sample_1, sample_2, alpha):
    """
    Two sided paired t-test
    H0: mean of the differences (sample_1 - sample_2) = 0
    Ha: mean of the differences != 0
    :return: t stat, p-value, and True/False for rejecting H0
    """
    # ttest_rel matches up each row of the two samples, takes the differences, and tests if the
    # mean difference is 0. using rel (paired) because both values come from the same jump
    (t_stat, p_value) = ttest_rel(sample_1, sample_2, alternative='two-sided')
    # return t, p, and whether p < alpha (True = reject H0)
    return t_stat, p_value, p_value < alpha


def plot_paired_comparison(x_values, y_values, x_label, y_label, title, file_name):
    """
    Scatter plot of accelerometer RSI vs force plate RSI for each trial.
    If a point is on the y = x line the two methods gave the exact same value.
    """
    # smallest and biggest values out of both columns
    data_min = min(x_values.min(), y_values.min())
    data_max = max(x_values.max(), y_values.max())
    # 5% of the data range as padding so the points aren't right on the edge
    # (using the range instead of multiplying by 0.95/1.05 so it also works if values are negative)
    pad = 0.05 * (data_max - data_min)
    # bottom and top of the axes
    low = data_min - pad
    high = data_max + pad

    # new square figure so both axes are the same scale
    plt.figure(figsize=(6, 6))
    # one dot per trial, force plate on x and accelerometer on y
    plt.scatter(x_values, y_values, color='navy', alpha=0.7, label='Trials (n = ' + str(len(x_values)) + ')')
    # dashed y = x line to show where the methods would agree perfectly
    plt.plot([low, high], [low, high], 'k--', label='Perfect agreement (y = x)')
    # title
    plt.title(title)
    # x label
    plt.xlabel(x_label)
    # y label
    plt.ylabel(y_label)
    # same limits on both axes so the y = x line goes corner to corner
    plt.xlim(low, high)
    plt.ylim(low, high)
    # legend
    plt.legend()
    # light grid
    plt.grid(alpha=0.3)
    # fixes spacing so labels don't get cut off
    plt.tight_layout()
    # save the plot as a png
    plt.savefig(file_name, dpi=200)


def expected_normal_counts(bins, mu, std, num_samples):
    """
    How many samples should be in each bin if the data really came from a normal dist with mu and std.
    """
    # cdf at each edge = chance of being below that edge. np.diff subtracts each edge from the next one
    # so you get the chance of landing inside each bin
    expected_prob = np.diff(norm.cdf(bins, loc=mu, scale=std))
    # chance times total number of samples = expected count
    return expected_prob * num_samples


def merge_small_bins(bins, observed_counts, expected_counts, min_expected):
    """
    Combines neighboring bins until every bin has an expected count of at least min_expected.
    The chi2 test isn't reliable when expected counts are too small (rule of thumb is 5).
    Merging two bins just deletes the edge between them and adds their counts together.
    :return: new bin edges, observed counts, expected counts (as numpy arrays)
    """
    # copy everything into lists so items can be deleted
    bins = list(bins)
    observed = list(observed_counts)
    expected = list(expected_counts)

    # keep merging while the smallest expected count is too small (and there's still more than 1 bin)
    while len(expected) > 1 and min(expected) < min_expected:
        # index of the bin with the smallest expected count
        i = int(np.argmin(expected))
        # pick which neighbor to merge with. the first bin can only merge right, the last bin can only
        # merge left, otherwise merge with whichever neighbor has the smaller expected count
        if i == 0:
            j = 1
        elif i == len(expected) - 1:
            j = i - 1
        elif expected[i - 1] < expected[i + 1]:
            j = i - 1
        else:
            j = i + 1
        # the left bin of the pair is the one that gets kept
        left = min(i, j)
        # add the right bin's counts into the left bin
        observed[left] = observed[left] + observed[left + 1]
        expected[left] = expected[left] + expected[left + 1]
        # delete the right bin
        del observed[left + 1]
        del expected[left + 1]
        # delete the edge that was between the two bins
        del bins[left + 1]

    # give back numpy arrays so the rest of the code works the same as before
    return np.array(bins), np.array(observed), np.array(expected)


def normal_fit_test(samples, alpha):
    """
    Fits a normal dist to the samples and does a chi2 goodness of fit test (same way as the module 5 example,
    plus merging bins that have expected counts under MIN_EXPECTED)
    H0: the data comes from the fitted normal dist
    Ha: the data doesn't come from the fitted normal dist
    :return: dictionary with the fit, bins, counts, chi2 stat, p-value, and whether it's a good fit
    """
    # step 1 - get the mean and std of the normal curve that fits the data best
    (mu, std) = norm.fit(samples)
    # if every value is the same the std is 0 and there's no normal curve to fit
    if std == 0:
        print('All values are identical (std = 0), so a normal distribution can\'t be fit.')
        sys.exit(-1)

    # step 2 - 9 edges from mu - 2std to mu + 2std, which makes 8 bins that are each half a std wide
    bins = np.linspace(mu - 2 * std, mu + 2 * std, 9)
    # add -inf and +inf on the ends so every value lands in a bin (10 bins now)
    bins = np.r_[-np.inf, bins, np.inf]

    # step 3 - count how many values are actually in each bin (don't need the edges it returns so used _)
    (observed_counts, _) = np.histogram(samples, bins=bins)

    # step 4 - how many the normal dist says should be in each bin
    expected_counts = expected_normal_counts(bins, mu, std, len(samples))

    # step 5 - merge bins with expected counts under MIN_EXPECTED. this usually happens in the tails.
    # keep the original bins too so they can be printed for comparison
    original_bins = bins
    original_observed = observed_counts
    original_expected = expected_counts
    (bins, observed_counts, expected_counts) = merge_small_bins(bins, observed_counts, expected_counts,
                                                                MIN_EXPECTED)

    # step 6 - dof = (number of bins after merging - 1) - 2, minus 2 since I estimated mu and std from the data
    dof = len(observed_counts) - 1 - 2
    # need at least 1 dof for the test to mean anything (happens if there are too few samples for 4+ bins)
    if dof < 1:
        print('Not enough data for a chi2 test after merging bins (dof = %d).' % dof)
        sys.exit(-1)

    # step 7 - chi2 test, adds up (O - E)^2 / E for all the merged bins. ddof=2 for the 2 estimated parameters
    (chi_stat, p_value) = chisquare(f_obs=observed_counts, f_exp=expected_counts, ddof=2)

    # put everything in a dictionary so main can print it and plot it
    # good_fit is True if p >= alpha (fail to reject H0)
    return {'mu': mu, 'std': std, 'bins': bins, 'observed': observed_counts, 'expected': expected_counts,
            'original_bins': original_bins, 'original_observed': original_observed,
            'original_expected': original_expected,
            'chi_stat': chi_stat, 'p_value': p_value, 'dof': dof,
            'good_fit': p_value >= alpha}


def plot_normal_fit(samples, fit, title, x_label, file_name):
    """
    Histogram of the data (as a density) with the fitted normal curve on top.
    The bars use the original half-std bins so the shape is easy to see, and dashed lines show
    the edges of the merged bins that the chi2 test actually used.
    """
    # original bin edges without the -inf and +inf, can't plot infinite bins
    finite_edges = fit['original_bins'][1:-1]
    # width of one bin (half a std)
    width = finite_edges[1] - finite_edges[0]
    # if any data is to the left of the first edge, add more bins on the left (same width) to cover it
    lower = min(finite_edges[0], finite_edges[0] - width * np.ceil((finite_edges[0] - samples.min()) / width))
    # same thing for the right side
    upper = max(finite_edges[-1], finite_edges[-1] + width * np.ceil((samples.max() - finite_edges[-1]) / width))
    # all the edges to actually plot, spaced one bin width apart
    plot_edges = np.arange(lower, upper + width / 2, width)

    # 300 x points so the curve looks smooth
    x = np.linspace(plot_edges[0], plot_edges[-1], 300)

    # new figure
    plt.figure(figsize=(8, 5))
    # histogram - density=True makes the total area 1 so it's on the same scale as the normal curve
    plt.hist(samples, bins=plot_edges, density=True, color='lightsteelblue', edgecolor='black',
             label='Accelerometer RSI (n = ' + str(len(samples)) + ')')
    # normal curve (pdf) in red on top of the histogram
    plt.plot(x, norm.pdf(x, loc=fit['mu'], scale=fit['std']), 'r-', linewidth=2,
             label='Fitted normal: μ = %.3f, σ = %.3f' % (fit['mu'], fit['std']))
    # finite edges of the bins the chi2 test used after merging (the outer bins go to +/- infinity)
    test_edges = fit['bins'][1:-1]
    # draw each test bin edge as a dashed gray line. only the first one gets a legend label
    for k, edge in enumerate(test_edges):
        plt.axvline(edge, color='gray', linestyle='--', linewidth=1,
                    label='Chi2 test bin edges (%d bins)' % len(fit['observed']) if k == 0 else None)
    # title
    plt.title(title)
    # x label
    plt.xlabel(x_label)
    # y label
    plt.ylabel('Probability density')
    # legend
    plt.legend()
    # light grid
    plt.grid(alpha=0.3)
    # fix spacing
    plt.tight_layout()
    # save as png
    plt.savefig(file_name, dpi=200)


def mean_confidence_interval(samples, confidence):
    """
    Confidence interval on the mean using the t dist (small sample and we don't know the real std)
    CI = mean +/- t_crit * (s / sqrt(n)) with n - 1 dof
    :return: mean, lower bound, upper bound
    """
    # mean of the samples
    mean = np.mean(samples)
    # t.interval gives mean -/+ t_crit * standard error. sem() is the standard error (s / sqrt(n)),
    # and dof is n - 1
    (lower, upper) = t.interval(confidence=confidence, df=len(samples) - 1, loc=mean, scale=sem(samples))
    # return the mean and both ends of the CI
    return mean, lower, upper


def plot_confidence_intervals(intervals, title, file_name):
    """Plots the mean for each method with its own CI as error bars"""
    # method names, these go on the x axis
    labels = list(intervals.keys())
    # mean for each method (spot 0 in each tuple)
    means = np.array([intervals[k][0] for k in labels])
    # lower bound for each method (spot 1)
    lowers = np.array([intervals[k][1] for k in labels])
    # upper bound for each method (spot 2)
    uppers = np.array([intervals[k][2] for k in labels])

    # new figure
    plt.figure(figsize=(6, 5))
    # dot at each mean with error bars going down to the lower bound and up to the upper bound.
    # the label says these are separate CIs for each method, not a test of the difference
    plt.errorbar(labels, means, yerr=[means - lowers, uppers - means], fmt='o', capsize=10, markersize=8,
                 color='navy', label='Mean RSI with ' + str(int(round(CONFIDENCE * 100))) + '% CI (each method)')
    # title
    plt.title(title)
    # x label
    plt.xlabel('Measurement method')
    # y label
    plt.ylabel('Mean RSI')
    # add some space on the sides so the error bars aren't right on the edge
    plt.xlim(-0.5, len(labels) - 0.5)
    # legend
    plt.legend()
    # light grid
    plt.grid(alpha=0.3)
    # fix spacing
    plt.tight_layout()
    # save as png
    plt.savefig(file_name, dpi=200)


# only runs if this file is run directly
if __name__ == '__main__':
    # load the data
    rsi = load_rsi_data(DATA_PATH)
    # print how many trials and participants there are
    print('Loaded', len(rsi), 'paired trials from', rsi['participant'].nunique(), 'participants.\n')

    # =================================================================================================================
    # TASK 1 - paired t-test, force plate vs accelerometer, all trials
    # =================================================================================================================
    # header for this section
    print('=' * 80)
    print('TASK 1: Paired t-test, force plate RSI vs accelerometer RSI (all trials)')
    print('=' * 80)
    # difference for each trial (force plate - accelerometer)
    differences = rsi[FP_COL] - rsi[ACC_COL]
    # print H0 and Ha
    print('H0: mean difference (force plate - accelerometer) = 0  -> the methods agree on average')
    print('Ha: mean difference (force plate - accelerometer) != 0 -> the methods disagree on average')
    # print number of pairs and the mean of each method
    print('Number of pairs: %d   Mean force plate RSI: %.4f   Mean accelerometer RSI: %.4f'
          % (len(rsi), rsi[FP_COL].mean(), rsi[ACC_COL].mean()))
    # print the mean and std of the differences
    print('Mean difference: %.4f   Std of differences: %.4f' % (differences.mean(), differences.std()))

    # do the paired t-test
    (t_stat, p_val, reject) = paired_t_test(rsi[FP_COL], rsi[ACC_COL], ALPHA)
    # print t and its degrees of freedom (number of pairs - 1)
    print('t-value: %.4f   (degrees of freedom = %d)' % (t_stat, len(rsi) - 1))
    # print p in scientific notation since it could be really small
    print('p-value: %.4e' % p_val)
    # p < alpha so reject H0
    if reject:
        # print the decision
        print('p < %.2f: REJECT the null hypothesis. The two methods give significantly different mean RSI;'
              % ALPHA)
        # if the mean difference is negative the accelerometer number is bigger, so it reads higher
        higher = 'accelerometer' if differences.mean() < 0 else 'force plate'
        # print which one reads higher
        print('          the %s reads higher on average.' % higher)
    # p >= alpha so fail to reject
    else:
        # print the decision
        print('p >= %.2f: FAIL TO REJECT the null hypothesis. No significant difference between methods.' % ALPHA)

    # limitation: the trial-level test treats all trials as independent, but each participant did several
    # trials, so trials from the same person are related. as a supplementary check (not a replacement),
    # average each participant's trials and redo the paired t-test with one pair per participant
    print('\nNote: the test above treats every trial as independent, but trials are grouped within')
    print('participants (trials per participant: %d to %d), so the trial-level p-value may be too small.'
          % (rsi['participant'].value_counts().min(), rsi['participant'].value_counts().max()))
    # mean force plate and accelerometer rsi for each participant
    participant_means = rsi.groupby('participant')[[FP_COL, ACC_COL]].mean()
    # need at least 2 participants for this check
    if len(participant_means) >= 2:
        # paired t-test on the participant averages
        (t_part, p_part, reject_part) = paired_t_test(participant_means[FP_COL], participant_means[ACC_COL], ALPHA)
        # print the result of the supplementary check
        print('Supplementary participant-level paired t-test (n = %d participants): t = %.4f, p = %.4e -> %s'
              % (len(participant_means), t_part, p_part,
                 'significant difference' if reject_part else 'no significant difference'))
    # only 1 participant so the check can't be done
    else:
        print('Only 1 participant, so the participant-level check can\'t be done.')

    # make the scatter plot, with the t-test result in the title so the plot matches the analysis
    plot_paired_comparison(rsi[FP_COL], rsi[ACC_COL], 'Force plate RSI', 'Accelerometer RSI',
                           'Paired RSI Measurements: Accelerometer vs Force Plate\n'
                           'Paired t-test: t = %.2f, p = %.2e' % (t_stat, p_val), 'task1_paired_comparison.png')
    # say where it saved
    print('Paired comparison plot saved as task1_paired_comparison.png')

    # =================================================================================================================
    # TASK 2 - normal fit and chi2 goodness of fit on the accelerometer RSI
    # =================================================================================================================
    # header for this section
    print('\n' + '=' * 80)
    print('TASK 2: Normal distribution fit and Chi2 goodness of fit, accelerometer RSI')
    print('=' * 80)
    # accelerometer column as a numpy array
    acc_values = rsi[ACC_COL].to_numpy()
    # fit the normal dist and run the chi2 test
    fit = normal_fit_test(acc_values, ALPHA)
    # print H0 and Ha
    print('H0: accelerometer RSI follows the fitted normal distribution')
    print('Ha: accelerometer RSI does not follow the fitted normal distribution')
    # print the fitted mean and std
    print('Fitted normal: mean = %.4f, std = %.4f' % (fit['mu'], fit['std']))

    # table of the original half-std bins before merging, so you can see which ones were too small
    original_table = pd.DataFrame({'bin lower': fit['original_bins'][:-1], 'bin upper': fit['original_bins'][1:],
                                   'observed': fit['original_observed'], 'expected': fit['original_expected']})
    # print the original bins
    print('Original bins (before merging):')
    print(original_table.round(4).to_string(index=False))
    # count how many original bins had expected counts under MIN_EXPECTED
    n_small = int((fit['original_expected'] < MIN_EXPECTED).sum())
    # print how many bins were merged away
    print('%d bin(s) had expected count < %d, so neighboring bins were merged: %d bins -> %d bins.'
          % (n_small, MIN_EXPECTED, len(fit['original_observed']), len(fit['observed'])))

    # table of each bin's edges, observed count, and expected count (these are the bins the test used)
    bin_table = pd.DataFrame({'bin lower': fit['bins'][:-1], 'bin upper': fit['bins'][1:],
                              'observed': fit['observed'], 'expected': fit['expected']})
    # how much each bin adds to the chi2 stat, (O - E)^2 / E. helps show which bins are off the most
    bin_table['(O-E)^2/E'] = (bin_table['observed'] - bin_table['expected']) ** 2 / bin_table['expected']
    # print the table rounded to 4 decimals without the row numbers
    print('Bins used in the chi2 test (after merging):')
    print(bin_table.round(4).to_string(index=False))

    # print chi2 stat and dof
    print('Chi2 stat: %.4f   (degrees of freedom = %d bins - 1 - 2 = %d)'
          % (fit['chi_stat'], len(fit['observed']), fit['dof']))
    # print p
    print('p-value: %.4e' % fit['p_value'])
    # p >= alpha means good fit
    if fit['good_fit']:
        # print that it's a good fit (failing to reject doesn't prove the data is normal, just that
        # the test didn't find evidence against it)
        print('p >= %.2f: FAIL TO REJECT H0. The normal distribution is a GOOD fit '
              '(no evidence against normality).' % ALPHA)
    # p < alpha means not a good fit
    else:
        # print that it's not a good fit
        print('p < %.2f: REJECT H0. The normal distribution is NOT a good fit.' % ALPHA)

    # make the histogram with the normal curve, with the chi2 result in the title
    plot_normal_fit(acc_values, fit, 'Accelerometer RSI: Histogram with Fitted Normal Distribution\n'
                    'Chi2 = %.2f, dof = %d, p = %.3f' % (fit['chi_stat'], fit['dof'], fit['p_value']),
                    'Reactive Strength Index (RSI)', 'task2_accel_normal_fit.png')
    # say where it saved
    print('Histogram saved as task2_accel_normal_fit.png')

    # =================================================================================================================
    # TASK 3 - participant F, 95% CIs for both methods
    # =================================================================================================================
    # header for this section
    print('\n' + '=' * 80)
    print('TASK 3: Participant %s, %d%% confidence intervals on mean RSI' % (PARTICIPANT, round(CONFIDENCE * 100)))
    print('=' * 80)
    # only keep rows for the chosen participant and just the trial + rsi columns, then reset the index
    participant_df = rsi.loc[rsi['participant'] == PARTICIPANT, [TRIAL_COL, FP_COL, ACC_COL]].reset_index(drop=True)
    # if nothing matched then that participant isn't in the data
    if participant_df.empty:
        # say so
        print('No trials found for participant ' + PARTICIPANT + '.')
        # quit
        sys.exit(-1)
    # a CI needs at least 2 trials (with 1 trial there's no std and dof would be 0)
    if len(participant_df) < 2:
        print('Participant ' + PARTICIPANT + ' only has 1 trial, so a confidence interval can\'t be found.')
        sys.exit(-1)
    # print the participant's dataframe
    print(participant_df)

    # mean and CI for each method, saved in a dictionary with the method name as the key
    intervals = {'Force plate': mean_confidence_interval(participant_df[FP_COL], CONFIDENCE),
                 'Accelerometer': mean_confidence_interval(participant_df[ACC_COL], CONFIDENCE)}
    # go through both methods and unpack (mean, lower, upper)
    for method, (mean, lower, upper) in intervals.items():
        # print mean, CI, and half width (distance from the mean to either end)
        print('%-13s: mean = %.4f, %d%% CI = (%.4f, %.4f), half-width = %.4f'
              % (method, mean, round(CONFIDENCE * 100), lower, upper, (upper - lower) / 2))

    # each CI above only describes where that method's own mean RSI probably is for this participant.
    # checking whether two separate CIs overlap is NOT a valid test of whether the methods differ, and
    # it ignores that both methods measured the same jumps. the right comparison for paired data is a CI
    # on the mean of the paired differences, so that's calculated here as a supporting check
    print('\nNote: the two CIs above are separate estimates for each method. Whether they overlap is not')
    print('a valid test of a difference, so the paired differences are used to compare the methods instead.')
    # difference for each of this participant's trials (force plate - accelerometer)
    participant_diffs = participant_df[FP_COL] - participant_df[ACC_COL]
    # CI on the mean difference
    (diff_mean, diff_lo, diff_hi) = mean_confidence_interval(participant_diffs, CONFIDENCE)
    # print the mean difference and its CI
    print('Mean paired difference (force plate - accelerometer) = %.4f, %d%% CI = (%.4f, %.4f)'
          % (diff_mean, round(CONFIDENCE * 100), diff_lo, diff_hi))
    # if 0 is inside the CI, a mean difference of 0 is still believable
    if diff_lo <= 0 <= diff_hi:
        # no significant difference found, but that doesn't prove the methods are the same
        print('The CI includes 0: no statistically significant difference was detected for participant %s'
              % PARTICIPANT)
        print('(this does not prove the methods agree; with n = %d trials the test has limited power).'
              % len(participant_df))
    # 0 is outside the CI
    else:
        # the methods differ for this participant
        print('The CI does not include 0: the methods give significantly different mean RSI for participant %s.'
              % PARTICIPANT)

    # make the CI plot
    plot_confidence_intervals(intervals, 'Participant %s: Mean RSI with %d%% Confidence Intervals'
                              % (PARTICIPANT, round(CONFIDENCE * 100)), 'task3_participant_ci.png')
    # say where it saved
    print('Confidence interval plot saved as task3_participant_ci.png')

    # show all the plots
    plt.show()
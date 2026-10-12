import sys

"""
Create a program which will provide answers to the questions posed in the assignment description.
We've provided a function which will parse the NYT covid database file (named "us-counties.csv"); 
however, its correct implementation will be up to you. DO NOT MODIFY THIS FUNCTION.
Your code needs to be successful as well as sufficiently commented/documented to receive full credit.
"""


def parse_nyt_data(file_path=''):
    """
    Parse the NYT covid database and return a list of tuples. Each tuple describes one entry in the source data set.
    Date: the day on which the record was taken in YYYY-MM-DD format
    County: the county name within the State
    State: the US state for the entry
    Cases: the cumulative number of COVID-19 cases reported in that locality
    Deaths: the cumulative number of COVID-19 death in the locality

    :param file_path: Path to data file
    :return: A List of tuples containing (date,county, state, fips, cases, deaths) information

    ____________________ DO NOT MODIFY THIS FUNCTION ___________________
    """
    # data point list
    data=[]

    # open the NYT file path
    try:
        fin = open(file_path)
    except FileNotFoundError:
        print('File ', file_path, ' not found. Exiting!')
        sys.exit(-1)

    # get rid of the headers
    fin.readline()

    # while not done parsing file
    done = False

    # loop and read file
    while not done:
        line = fin.readline()

        if line == '':
            done = True
            continue

        # format is date,county,state,fips,cases,deaths
        (date,county, state, fips, cases, deaths) = line.rstrip().split(",")

        # clean up the data to remove empty entries
        if cases=='':
            cases=0
        if deaths=='':
            deaths=0

        # convert elements into ints
        try:
            entry = (date,county,state, fips, int(cases), int(deaths))
        except ValueError:
            print('Invalid parse of ', entry)

        # place entries as tuple into list
        data.append(entry)


    return data

### YOUR CODE HERE ###

import os  ## os lets me build a file path that works no matter where the script is run from
from datetime import date, timedelta  ## date/timedelta let me do real calendar math (add one day at a time)


## Build the path from this script's own location: go up two folders to the repo root, then into data/covid.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  ## folder that this script is saved in
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'covid', 'us-counties.csv')  ## full path to the NYT csv


def get_cumulative_series(data, county_name, state_name):
    """
    Build a continuous day-by-day series of CUMULATIVE case counts for one locality.

    The NYT file only has a row for a locality on days it was reported, so any missing calendar
    days are filled by carrying the previous cumulative value forward (zero new cases on those days).
    This keeps the 7-day windows in Question 3 true calendar weeks.

    :param data: list of tuples from parse_nyt_data
    :param county_name: exact county name as it appears in the file
    :param state_name: state name (needed because county names repeat across states)
    :return: (list of datetime.date, list of cumulative case counts), same length
    """
    reported = {}  ## dictionary mapping each reported date -> cumulative cases for this locality
    for (d, county, state, fips, cases, deaths) in data:  ## loop through every row in the whole data set
        if county == county_name and state == state_name:  ## keep only rows for the county AND state I want
            y, m, day = d.split('-')  ## split the 'YYYY-MM-DD' string into year, month, and day pieces
            reported[date(int(y), int(m), int(day))] = cases  ## store the cases under a real date object

    if not reported:  ## if nothing matched, the name or state was wrong (or missing from the file)
        return [], []  ## return empty lists so main() can detect the problem

    first_day = min(reported)  ## earliest date this locality shows up in the data
    last_day = max(reported)  ## latest date this locality shows up in the data

    dates = []  ## will hold every calendar day from first_day to last_day
    cumulative = []  ## will hold the cumulative case count for each of those days
    previous = 0  ## running total from the previous day (starts at 0 before any cases)
    current = first_day  ## the day I'm currently filling in, starting at the first reported day
    while current <= last_day:  ## step through every calendar day until the last reported day
        previous = reported.get(current, previous)  ## use the reported value if there is one, else keep yesterday's total
        dates.append(current)  ## record this day
        cumulative.append(previous)  ## record the cumulative total for this day
        current += timedelta(days=1)  ## move on to the next calendar day
    return dates, cumulative  ## give back the two parallel lists


def daily_new_cases(cumulative):
    """
    Convert cumulative totals to new cases per day (today's total minus yesterday's).
    The first day's new cases equal its cumulative total (nothing came before it).
    Note: NYT sometimes revises totals downward, which can make a day's value negative.
    That's fine here because the max calculations below just ignore those small negatives.
    """
    new = []  ## list that will hold the new cases for each day
    previous = 0  ## yesterday's cumulative total (0 before the first day)
    for total in cumulative:  ## go through each day's cumulative total in order
        new.append(total - previous)  ## new cases = today's total minus yesterday's total
        previous = total  ## today becomes "yesterday" for the next loop
    return new  


def first_positive_case(dates, cumulative):
    """Return the first date on which the cumulative case count is above zero."""
    for d, total in zip(dates, cumulative):  ## walk through each date alongside its cumulative total
        if total > 0:  ## the first time the total is above zero is the first positive case
            return d  
    return None  ## if the total never went above zero, there was no case


def max_single_day(dates, new_cases):
    """Return (date, count) for the day with the most new cases (earliest date wins a tie)."""
    best_index = 0  ## index of the best day found so far (start with the first day)
    for i in range(len(new_cases)):  ## check every day by its index
        if new_cases[i] > new_cases[best_index]:  ## strictly greater, so on a tie the earlier day stays
            best_index = i  ## this day is the new best
    return dates[best_index], new_cases[best_index]  ## return the date and how many new cases that day had


def worst_seven_day_period(dates, cumulative, window=7):
    """
    Find the 7-day window with the most new cases.

    New cases over days i-6 .. i equal cumulative[i] - cumulative[i-7] (using 0 as the total before
    the first day). This avoids re-adding seven numbers for every window. Only full 7-day windows
    are considered, and the earliest window wins a tie.

    :return: (start_date, end_date, total_new_cases)
    """
    padded = [0] + cumulative  ## put a 0 in front so padded[k+1] is cumulative[k] and padded[0] is the pre-outbreak total
    best_end = None  ## index of the last day of the best window so far
    best_total = None  ## new-case total of the best window so far
    for end in range(window - 1, len(dates)):  ## 'end' is the last day of the window; first full window ends at index 6
        total = padded[end + 1] - padded[end + 1 - window]  ## cumulative at the end minus cumulative just before the window starts
        if best_total is None or total > best_total:  ## first window, or strictly bigger than the best so far
            best_total = total  ## remember the new best total
            best_end = end  ## remember where that window ended
    return dates[best_end - window + 1], dates[best_end], best_total  ## start date, end date, and new cases in the window


def main():
    """Load the data, build a series for each locality, and print the answers to all three questions."""
    data = parse_nyt_data(DATA_FILE)  ## read the whole NYT file into a list of tuples

    ## Harrisonburg is an independent city, so some versions of the file call it "Harrisonburg city"
    ## and others call it "Harrisonburg". I check which spelling is actually in this file.
    names_in_file = {c for (_, c, s, _, _, _) in data if s == 'Virginia'}  ## set of every Virginia county/city name in the file
    if 'Harrisonburg city' in names_in_file:  ## this version of the file uses the "city" suffix
        harrisonburg_name = 'Harrisonburg city'  ## so use that spelling
    else:  
        harrisonburg_name = 'Harrisonburg'  ## use the other spelling

    ## Each entry is (label for printing, name in the NYT file, state).
    ## Rockingham must be filtered by state because North Carolina and New Hampshire also have a Rockingham County.
    localities = [
        ('Harrisonburg', harrisonburg_name, 'Virginia'),  ## the city of Harrisonburg
        ('Rockingham County', 'Rockingham', 'Virginia'),  ## Rockingham County, Virginia
    ]

    results = {}  ## maps each label -> (dates, cumulative, new cases per day)
    for label, county, state in localities:  ## process each locality one at a time
        dates, cumulative = get_cumulative_series(data, county, state)  ## build the day-by-day cumulative series
        if not dates:  ## empty result means the locality wasn't found in the file
            print('No data found for', label)  ## tell the user which one failed
            sys.exit(-1)  ## stop the program since the answers would be wrong
        results[label] = (dates, cumulative, daily_new_cases(cumulative))  ## store the series plus the daily new cases

    print('QUESTION 1: First positive COVID case')  
    for label in results:  ## go through each locality
        dates, cumulative, new = results[label]  ## unpack its stored data
        print(f'  {label}: {first_positive_case(dates, cumulative)}')  ## print the first date with a positive case

    print('\nQUESTION 2: Day with the greatest number of new cases')  
    for label in results:  ## go through each locality
        dates, cumulative, new = results[label]  ## unpack its stored data
        d, n = max_single_day(dates, new)  ## find the worst single day and its new-case count
        print(f'  {label}: {d} ({n} new cases)')  ## print the date and the count

    print('\nQUESTION 3: Worst seven-day period for new cases')  
    for label in results:  ## go through each locality
        dates, cumulative, new = results[label]  ## unpack its stored data
        start, end, total = worst_seven_day_period(dates, cumulative)  ## find the 7-day window with the most new cases
        print(f'  {label}: {start} to {end} ({total} new cases)')  ## print the window's start, end, and total

if __name__ == '__main__':  ## only run main() when this file is run directly, not when imported
    main()  ## start the program
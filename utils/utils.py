import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def simple_matchup_projection(df, home_team, away_team, stats=None):
    '''
    Give the projected score of two teams and their projected stats
    based on their *total stats* of the season so far.
    '''
    # select the two teams' data from the stat table
    if stats is None:
        dftemp = df.loc[[home_team, away_team], :].T
    else:
        dftemp = df.loc[[home_team, away_team], stats].T
    # find the TO (turnovers) index (TO: the lower the better)
    to = dftemp.index == 'TO'
    # take the difference between the two teams (away - home)
    stat_diff = np.diff(dftemp.values, axis=1)
    # when the diff is positive assign it a 2 (win of away),
    # otherwise a 1 (win of home)
    winner = np.where(stat_diff > 0, 2, 1)
    # correct for turnovers (see above)
    winner[to] = 2 if winner[to] == 1 else 1
    # correct for ties
    winner = np.where(stat_diff == 0, 0, winner)

    # Replace 1s and 2s with the teams abbreviation.
    dftemp['Projected Winner'] = np.where(
        winner == 0, None, np.where(winner == 1, home_team, away_team))

    print('projected score %s-%s: %d-%d-%d' % (home_team, away_team,
                                               (winner == 1).sum(),
                                               (winner == 2).sum(),
                                               (winner == 0).sum()))
    return dftemp.round(4).T


def fantasy_team_schedule_count(fantasy_team_data, nba_schedule_data, week,
                                fantasy_team_abbr, n_active_players):
    '''
    Returns the number of games played each week by a fantasy team with the
    *current* roster.
    '''
    # find the index of the fantasy team of interest
    team_index = next((i for i, u in enumerate(fantasy_team_data['teams'])
                       if u['abbrev'] == fantasy_team_abbr), None)

    if team_index is None:
        print('Error: Fantasy team not found')
        return None
    # isolate the team dictionary
    team = fantasy_team_data['teams'][team_index]
    # find the team ids of the players of the fantasy team.
    teams_playing = [player['playerPoolEntry']['player']['proTeamId']
                     for player in team['roster']['entries']
                     if not player['playerPoolEntry']['player']['injured']]
    # find the number of players in the same nba team
    uniq, counts = np.unique(teams_playing, return_counts=True)
    # make a dictionary with keys the nba team ids and values the number of
    # players of the fantasy team in the nba team
    teams_dict = dict(zip(uniq, counts))
    # boolean mask of teams playing in the week of interest.
    ii = ((nba_schedule_data.Week == week)
          & (nba_schedule_data.Home_id.isin(teams_playing)
             | nba_schedule_data.Away_id.isin(teams_playing)))
    # filter the data
    temp = nba_schedule_data.loc[ii, ['Date', 'Home_id', 'Away_id']].copy()
    # filter columns
    temp2 = temp[['Home_id', 'Away_id']].copy()
    # boolean mask of the teams that are not playing for the fantasy team
    bb = ~(temp2.isin(teams_dict.keys())).values
    # replace the teams that are not playing for the fantasy team with 0.
    temp2[bb] = 0
    # replace all other teams (those that play for the fantasy team) with the
    # number of player that they play for
    temp3 = temp2.replace(teams_dict).sum(axis=1).to_frame('Count')
    # group by date and count
    final_table = (pd.concat((temp, temp3), axis=1)
                   .groupby(['Date'])[['Count']].sum())
    # make valid columns (only 9 are available)
    final_table['Count_valid'] = np.where(
        final_table['Count'] > n_active_players, n_active_players,
        final_table['Count'])
    # make new column to show the unsused subs
    final_table['Unused_subs'] = np.where(
        final_table['Count'] > n_active_players,
        final_table['Count'] - n_active_players, 0)
    return final_table


def compare_schedules(fantasy_team_data, nba_schedule_data, week,
                      fantasy_team1, fantasy_team2, n_active_players):
    '''
    Compares schedule of a given week of two fantasy teams
    '''
    # get the schedule table of the first team
    table1 = fantasy_team_schedule_count(fantasy_team_data, nba_schedule_data,
                                         week, fantasy_team1, n_active_players)
    # get the schedule table of the second team
    table2 = fantasy_team_schedule_count(fantasy_team_data, nba_schedule_data,
                                         week, fantasy_team2, n_active_players)
    # merge the two on the dates
    table = table1.merge(table2, how='outer', right_index=True, left_index=True,
                         suffixes=('-%s' % fantasy_team1,
                                   '-%s' % fantasy_team2))
    # return the total valid users too.
    total = table.loc[:, table.columns.str.startswith('Count_valid')].sum()
    return table, total


def fantasy_team_ranking_per_stat(table):
    '''
    Ranking index for each statistical category.
    '''
    ranks_df = table.copy()
    for stat in ranks_df.columns:
        if stat == 'TO':
            ranks_df[stat] = table[stat].argsort().argsort().values + 1
        else:
            ranks_df[stat] = table[stat].argsort()[::-1].argsort().values + 1
    return ranks_df


def filter_by_round_stats(table, week, stats):
    '''
    Filters stats by round (`week`)
    '''
    table.set_index('teamAbbr', inplace=True)
    table = table.loc[table['Round'] == week, stats]
    return table


def fantasy_matchup_score(x, y, index_to=None):
    '''
    Computes the fantasy score, i.e. the difference between the win
    and losses in a match-up
    '''
    # take the difference between the stats of the two teams
    diff = x - y
    # take the sign of the differences
    wins = np.sign(diff)
    # correct for TO
    if index_to is not None:
        wins[index_to] = - wins[index_to]
    # return the sum of the wins-losses
    return wins.sum()


def h2h_score_matrix(stat_table):
    '''
    Computes the score matrix, i.e. the match-up score between
    all teams against all other teams, for a given week
    '''
    # get the team abbreviations for the filtered dataframe.
    teams = stat_table.index
    # get the stat values of
    X = stat_table.values
    # find the index of the 'TO' (to correct)
    index_to = np.where(stat_table.columns == 'TO')[0][0]

    # temporary metric function using the `fantasy_matchup_score`
    # and the `index_to`
    metric = lambda x, y: fantasy_matchup_score(x, y, index_to)
    # make the square scores matrix
    dist_mat = squareform(pdist(X, metric=metric))
    # the metric function is symmetric, correct it and make it anti-symmetric
    # (if team A wins team B, e.g. +2, team B loses from team A, i.e. -2)
    dist_mat[np.tril_indices(dist_mat.shape[0])] = (
        - dist_mat[np.tril_indices(dist_mat.shape[0])])
    # make the scores matrix into a dataframe
    df = pd.DataFrame(dist_mat, index=teams, columns=teams).astype(int)
    return df


def aggregate_h2h_scores(score_mat):
    '''
    Computes aggregates scores for each fantasy team from
    their performance against every other team
    '''
    # get the number of wins against all other teams
    wins = (score_mat > 0).sum(axis=1)
    # get the percentage of wins
    perc = wins / (score_mat.shape[0] - 1)
    # get the average score against all other teams
    avg = score_mat.sum(axis=1) / (score_mat.shape[0] - 1)
    # make a dataframe
    scores_df = pd.DataFrame({'Wins': wins, 'Wins%': perc, 'Avg Score': avg})
    return scores_df


def advanced_stats_by_fantasy_team(data, stat_codes):
    '''
    Fetches advanced stats (specified in `stat_codes`) for all fantasy teams
    in league.
    '''
    all_players_data = []
    for player_data in data['rosterForMatchupPeriod']['entries']:
        player = player_data['playerPoolEntry']['player']
        player_stats_dict = player['stats'][0]['stats']
        player_stats_dict['player_name'] = player['fullName']
        all_players_data.append(player_stats_dict)
    out_df = (pd.DataFrame(all_players_data).set_index('player_name')
              .loc[:, stat_codes].sum(axis=0))
    return out_df

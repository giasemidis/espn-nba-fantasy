import numpy as np
import pandas as pd
from IPython import embed
# from scipy.spatial.distance import pdist
# from scipy.spatial.distance import squareform


# def simple_matchup_projection(df, home_team, away_team, stats=None):
#     '''
#     Give the projected score of two teams and their projected stats
#     based on their *total stats* of the season so far.
#     '''
#     # select the two teams' data from the stat table
#     if stats is None:
#         dftemp = df.loc[[home_team, away_team], :].T
#     else:
#         dftemp = df.loc[[home_team, away_team], stats].T
#     # find the TO (turnovers) index (TO: the lower the better)
#     to = dftemp.index == 'TO'
#     # take the difference between the two teams (away - home)
#     stat_diff = np.diff(dftemp.values, axis=1)
#     # when the diff is positive assign it a 2 (win of away),
#     # otherwise a 1 (win of home)
#     winner = np.where(stat_diff > 0, 2, 1)
#     # correct for turnovers (see above)
#     winner[to] = 2 if winner[to] == 1 else 1
#     # correct for ties
#     winner = np.where(stat_diff == 0, 0, winner)
#
#     # Replace 1s and 2s with the teams abbreviation.
#     dftemp['Projected Winner'] = np.where(
#         winner == 0, None, np.where(winner == 1, home_team, away_team))
#
#     print('projected score %s-%s: %d-%d-%d' % (home_team, away_team,
#                                                (winner == 1).sum(),
#                                                (winner == 2).sum(),
#                                                (winner == 0).sum()))
#     return dftemp.round(4).T


# def fantasy_team_ranking_per_stat(table):
#     '''
#     Ranking index for each statistical category.
#     '''
#     ranks_df = table.copy()
#     for stat in ranks_df.columns:
#         if stat == 'TO':
#             ranks_df[stat] = table[stat].argsort().argsort().values + 1
#         else:
#             ranks_df[stat] = table[stat].argsort()[::-1].argsort().values + 1
#     return ranks_df

def matchup_stats(match, stat_cols, where='home'):
    '''
    TODO: figure out what this function does and improve naming
    '''
    matchperiodid = match['matchupPeriodId']
    there = match[where]
    if 'cumulativeScore' not in there.keys():
        return None
    there_cum = there['cumulativeScore']
    there_data = [there['teamId'], where, there_cum['wins'],
                  there_cum['losses'], there_cum['ties']]
    there_stat = [0 if there_cum['scoreByStat'] is None else
                  there_cum['scoreByStat'][stat]['score']
                  for stat in stat_cols]
    return [matchperiodid] + there_data + there_stat


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


# def h2h_score_matrix(stat_table):
#     '''
#     Computes the score matrix, i.e. the match-up score between
#     all teams against all other teams, for a given week
#     '''
#     # get the team abbreviations for the filtered dataframe.
#     teams = stat_table.index
#     # get the stat values of
#     X = stat_table.values
#     # find the index of the 'TO' (to correct)
#     index_to = np.where(stat_table.columns == 'TO')[0][0]
#
#     # temporary metric function using the `fantasy_matchup_score`
#     # and the `index_to`
#     metric = lambda x, y: fantasy_matchup_score(x, y, index_to)
#     # make the square scores matrix
#     dist_mat = squareform(pdist(X, metric=metric))
#     # the metric function is symmetric, correct it and make it anti-symmetric
#     # (if team A wins team B, e.g. +2, team B loses from team A, i.e. -2)
#     dist_mat[np.tril_indices(dist_mat.shape[0])] = (
#         - dist_mat[np.tril_indices(dist_mat.shape[0])])
#     # make the scores matrix into a dataframe
#     df = pd.DataFrame(dist_mat, index=teams, columns=teams).astype(int)
#     return df


# def aggregate_h2h_scores(score_mat):
#     '''
#     Computes aggregates scores for each fantasy team from
#     their performance against every other team
#     '''
#     # get the number of wins against all other teams
#     wins = (score_mat > 0).sum(axis=1)
#     # get the percentage of wins
#     perc = wins / (score_mat.shape[0] - 1)
#     # get the average score against all other teams
#     avg = score_mat.sum(axis=1) / (score_mat.shape[0] - 1)
#     # make a dataframe
#     scores_df = pd.DataFrame({'Wins': wins, 'Wins%': perc, 'Avg Score': avg})
#     return scores_df


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

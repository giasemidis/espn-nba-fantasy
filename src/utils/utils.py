import numpy as np
import pandas as pd
from utils.get_logger import get_logger

logger = get_logger(__name__)


def matchup_stats(match, stat_cols, where='home'):
    '''
    Extracts the specified statistics (`stat_cols) for the input fantasy team
    (`where`) for a particular match-up (`match`).

    `match` is a dictionary returned from the ESPN API.
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


def extract_player_stat(player_info, stat_type_code):
    """
    A fucntion that extracts the player's stats from the given stat type code.
    `player_info` is a dictionary as returned from the ESPN API.
    """
    player_name = player_info['fullName']
    player_stats = player_info['stats']
    injury_status = player_info['injuryStatus']
    pro_team_id = player_info['proTeamId']
    player_dict = {}
    for stat_type in player_stats:
        if stat_type['id'] == stat_type_code:
            if 'averageStats' not in stat_type:
                logger.warning(
                    f"Player {player_name} does not have requested data"
                )
                continue
            player_dict = stat_type['averageStats']
            player_dict['Name'] = player_name
            player_dict['injuryStatus'] = injury_status
            player_dict['proTeamId'] = pro_team_id
            break
    return player_dict


def fantasy_team_schedule_count(roster_schedule_df, n_active_players):
    """
    Counts the number of players available for each day of the match-up.
    It also returns the number of unused players based on the leagues' cap.
    Caution: It does not count for players not fitting becuase of their
    position, 8 guards.
    """
    # count number of games per day of the fantasy roster team
    groupby_df = (roster_schedule_df.groupby(['date']).count()['team']
                  .to_frame('Count'))
    # count valid/active players
    groupby_df['Count_valid'] = np.where(
        groupby_df['Count'] > n_active_players, n_active_players,
        groupby_df['Count'])
    # make new column to show the unsused subs
    groupby_df['Unused_subs'] = np.where(
        groupby_df['Count'] > n_active_players,
        groupby_df['Count'] - n_active_players, 0)
    return groupby_df


def filter_by_round_team_stats(table, week, stats, teams=[]):
    '''
    Filters stats by round (`week`)
    '''
    # table.set_index('teamAbbr', inplace=True)
    table = table.loc[table['Round'] == week, stats]
    if teams:
        table = table.loc[table.index.isin(teams)]
    return table


def estimate_made_shots(attempts, perc):
    """
    Estimate made shots from attempts (made cannot be greater than
    attempted shots) and the percentage of success.
    """
    made_shots = np.zeros(attempts.shape)
    for atmp in np.unique(attempts):
        if atmp == 0:
            continue
        rows, cols = np.where(attempts == atmp)
        r = np.random.uniform(size=(cols.shape[0], atmp))
        made = r <= perc[cols][:, None]
        made_shots[rows, cols] = made.sum(axis=-1)
    return made_shots


def simulate_schedule(team_schedule_stats_df, poisson_stats, n_reps):
    # poisson_stats = self.poisson_stats
    pois_stats_arr = (
        team_schedule_stats_df.loc[:, poisson_stats].fillna(0).values
    )

    poisson_samples = np.random.poisson(
        lam=pois_stats_arr, size=(n_reps, *pois_stats_arr.shape)
    )
    poisson_samples = np.maximum(0, poisson_samples)

    # Estimate made shots, bounded above from attempted shots
    fga = poisson_samples[..., poisson_stats.index('FGA')]
    fta = poisson_samples[..., poisson_stats.index('FTA')]

    fgm = estimate_made_shots(
        fga, team_schedule_stats_df.loc[:, 'FG%'].values
    )
    ftm = estimate_made_shots(
        fta, team_schedule_stats_df.loc[:, 'FT%'].values
    )

    assert (fgm <= fga).all(), "FGM error"
    assert (ftm <= fta).all(), "FTM error"

    # make sure the simulated stats array have the headers ordered
    # self.simulation_stats = self.poisson_stats + ['FGM', 'FTM']
    # simulation_stats = poisson_stats + ['FGM', 'FTM']
    # concatenate poisson_samples with FGM and FTM
    simulated_stats_arr = np.concatenate(
        (poisson_samples, fgm[..., None], ftm[..., None]), axis=-1
    )

    # aggregate across time (week round)
    aggr_pois_stats = simulated_stats_arr.sum(axis=1)

    return aggr_pois_stats


def shot_percentage(array, made_idx, attmp_idx):
    '''
    Return the shot percentage from made and attempted shot index.
    `array` must be of size `n_reps` x number of stats
    '''
    assert (array[:, made_idx] <= array[:, attmp_idx]).all(), (
        "made to attempts error"
    )
    return (array[:, made_idx] / array[:, attmp_idx])[:, None]

import pytz
# from datetime import datetime
import requests
import json
import numpy as np
import pandas as pd
from src.utils.utils import advanced_stats_by_fantasy_team


class EspnFantasyLeague():
    def __init__(self, league_id, season, n_active_players,
                 url_fantasy, url_nba, cookies, stat_type_code='002022'):
        self.league_id = league_id
        self.season = season
        self.cookies = cookies
        self.n_active_players = n_active_players
        self.url_fantasy = url_fantasy.format(season, league_id)
        self.url_nba = url_nba.format(season)

        self.dtypes = {'FG%': float, 'FT%': float, '3PM': int, 'REB': int,
                       'AST': int, 'STL': int, 'BLK': int, 'TO': int,
                       'PTS': int, 'FTA': int, 'FTM': int,
                       'FGA': int, 'FGM': int}
        self.stat_id_abbr_dict = {'0': 'PTS', '1': 'BLK', '2': 'STL',
                                  '3': 'AST', '6': 'REB', '11': 'TO',
                                  '13': 'FGM', '14': 'FGA',
                                  '15': 'FTM', '16': 'FTA',
                                  '17': '3PM',
                                  '19': 'FG%', '20': 'FT%'}
        self.adv_stats_dict = {'40': 'Mins', '42': 'Games'}
        self.fantasy_stats = ['FG%', 'FT%', '3PM', 'REB',
                              'AST', 'STL', 'BLK', 'TO', 'PTS']

        # stats needed for simulation
        self.stats_aux = ['FGM', 'FGA', 'FTM', 'FTA', 'FG%', 'FT%',
                          '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']
        # stats simulated with a poisson distribution
        self.poisson_stats = ['FGA', 'FTA', '3PM', 'REB', 'AST',
                              'STL', 'BLK', 'TO', 'PTS']
        self.simulation_stats = self.poisson_stats + ['FGM', 'FTM']

        self.team_id_abbr_dict = {}
        self.team_id_name_dict = {}
        self.team_abbr_name_dict = {}
        self.division_id_name_dict = {}
        self.n_teams = None
        self.stat_type_code = stat_type_code

        return

    def get_espn_data(self, url_endpoint, endpoints=[], headers=None, **kargs):
        '''
        Fetch data from the ESPN:
        For fantasy league data the available end-points are:
        * view=mDraftDetail
        * view=mLiveScoring
        * view=mMatchup
        * view=mPendingTransactions
        * view=mPositionalRatings
        * view=mSettings
        * view=mTeam
        * view=modular
        * view=mNav
        * view=mMatchupScore
        * view=mStandings
        * view=mRoster
        * view=kona_player_info
        '''
        params = {'view': endpoints, **kargs}
        r = requests.get(url_endpoint, cookies=self.cookies, params=params,
                         headers=headers)
        if r.status_code != 200:
            raise ValueError('Error fetching the teams data')
        data = r.json()
        return data

        '''
        Fetch fantasy teams' data
        '''
        data = self.get_espn_data(self.url_fantasy,
                                  endpoints=['mTeam', 'mRoster', 'mSettings',
                                             'mMatchup'])

        self.get_league_team_division_settings(data)
        return data

    def get_players_data(self):
        '''
        '''
        filters = {
            "players": {
                "filterStatus": {
                    "value": ["FREEAGENT", "WAIVERS"]
                },
                "limit": 5000,
                "sortDraftRanks": {
                    "sortPriority": 100,
                    "sortAsc": True,
                    "value": "STANDARD"
                }
            }
        }
        headers = {'x-fantasy-filter': json.dumps(filters)}
        data = self.get_espn_data(self.url_fantasy,
                                  endpoints=['kona_player_info'],
                                  headers=headers)
        return data

    def get_league_team_division_settings(self, data):
        '''
        Requires `mTeam` and `mSettings` endpoints
        '''
        self.team_id_abbr_dict = {team['id']: team['abbrev']
                                  for team in data['teams']}
        self.team_id_name_dict = {
            team['id']: team['location'] + ' ' + team['nickname']
            for team in data['teams']}
        self.team_abbr_name_dict = {
            v: self.team_id_name_dict[k]
            for k, v in self.team_id_abbr_dict.items()
        }
        self.division_id_name_dict = {
            u['id']: u['name']
            for u in data['settings']['scheduleSettings']['divisions']}
        if len(self.division_id_name_dict) > 1:
            print('There are more than 1 divisions')
        self.n_teams = len(data['teams'])
        print('%d teams participating' % self.n_teams)
        return

    # def make_stat_table(self, data):
    #     '''
    #     Makes the *current* total stat table
    #     It requires the `mTeam` endpoint data
    #     '''
    #     stats = [team['valuesByStat'] for team in data['teams']]
    #     team_abbrs = [team['abbrev'] for team in data['teams']]
    #     # make dataframe
    #     stat_table = pd.DataFrame(data=stats, index=team_abbrs)
    #     # select only headers of interest
    #     stat_table = stat_table.loc[:, self.stat_id_abbr_dict.keys()]
    #     # rename stat headers to match the web app
    #     stat_table.rename(columns=self.stat_id_abbr_dict, inplace=True)
    #     # update the column data types
    #     stat_table.astype(self.dtypes)
    #     return stat_table[self.fantasy_stats]
    #
    # def make_standings(self, data, division='all'):
    #     '''
    #     Makes the *current* standings table
    #     It requires the `mTeam` and `mSettings` endpoint data
    #     '''
    #     if len(self.division_id_name_dict) == 0:
    #         self.division_id_name_dict = {
    #             u['id']: u['name'] for u in
    #             data['settings']['scheduleSettings']['divisions']
    #         }
    #     team_abbrs = [team['abbrev'] for team in data['teams']]
    #     division_ids = [self.division_id_name_dict[team['divisionId']]
    #                     for team in data['teams']]
    #     record = [team['record']['overall'] for team in data['teams']]
    #     standings = pd.DataFrame(data=record, index=team_abbrs)
    #     standings = standings.loc[:, ['wins', 'losses', 'ties', 'percentage']]
    #     standings['Division'] = division_ids
    #
    #     if len(self.division_id_name_dict) > 1:
    #         if division.lower() == 'west':
    #             standings = standings[standings['Division'] == 'West']
    #         elif division.lower() == 'east':
    #             standings = standings[standings['Division'] == 'East']
    #
    #     standings = standings.sort_values('percentage', ascending=False)
    #     standings['Rank'] = np.arange(1, standings.shape[0] + 1, dtype=int)
    #     return standings
    #
    # def make_stat_standings_table(self, data):
    #     '''
    #     Makes the stat and standings table
    #     '''
    #     # make the stat table
    #     stat_table = self.make_stat_table(data)
    #     # make the standings table
    #     standings = self.make_standings(data)
    #     # merge the two on the indices
    #     table = standings.merge(stat_table, right_index=True, left_index=True)
    #     return table
    #
    # def get_total_team_stats_upto_round(self, data, week):
    #     '''
    #     Get the total statistics for each fantasy team from the beginning of
    #     season up to a specified week. If wee = current week, this function
    #     is equivalent to the `make_stat_table` method above.
    #
    #     It requires `mTeam`, `mRoster`, `mSettings`, `mMatchup`
    #     '''
    #     table = self.get_fantasy_team_stats_per_round(data)
    #     table = table[table['Round'] <= week]
    #     stats = list(self.stat_id_abbr_dict.values())
    #     aggregate = table.groupby('teamAbbr')[stats].sum()
    #     aggregate['FT%'] = aggregate['FTM'] / aggregate['FTA']
    #     aggregate['FG%'] = aggregate['FGM'] / aggregate['FGA']
    #     return aggregate[self.fantasy_stats]
    #
    def get_adv_stats_per_fantasy_team(self, endpoints=[],
                                       scoring_period=None):
        ''''''
        data = self.get_espn_data(self.url_fantasy, endpoints=endpoints,
                                  scoringPeriodId=scoring_period)
        data_df = pd.DataFrame([])
        matchupPeriodId = []
        stat_codes = list(self.adv_stats_dict.keys())
        for matchup in data['schedule']:
            if ('rosterForMatchupPeriod' in matchup['home']
                    and 'rosterForMatchupPeriod' in matchup['away']):
                matchupPeriodId.append(matchup['matchupPeriodId'])
                home_abbr = self.team_id_abbr_dict[matchup['home']['teamId']]
                away_abbr = self.team_id_abbr_dict[matchup['away']['teamId']]
                home_stats = advanced_stats_by_fantasy_team(matchup['home'],
                                                            stat_codes)
                away_stats = advanced_stats_by_fantasy_team(matchup['away'],
                                                            stat_codes)
                df = pd.concat((home_stats, away_stats), axis=1)
                df.rename(columns={0: home_abbr, 1: away_abbr}, inplace=True)
                data_df = pd.concat((data_df, df), axis=1)
        print('Processing matchup period:', np.unique(matchupPeriodId))
        data_df = data_df.T.rename(columns=self.adv_stats_dict)
        return data_df

    def make_team_schedule_table(self, data):
        '''
        Returns the NBA games (hence the teams) schedule for the whole season
        with additional info such as week, etc.
        '''
        us_central = pytz.timezone('US/Central')

        teams_data = []
        for team in data['settings']['proTeams']:
            # loop over all teams
            if team['abbrev'] == 'FA':
                # skip the free agents
                continue
            # this is the data for each game
            d = team['proGamesByScoringPeriod']
            for u in d:
                # loop over all games of the season for the team
                if d[u] != []:
                    tmp_data = [u, d[u][0]['date'], d[u][0]['homeProTeamId'],
                                d[u][0]['awayProTeamId']]
                    teams_data.append(tmp_data)
        # make dataframe from the data
        cols = ['Scoring Period', 'Date_int', 'Home_id', 'Away_id']
        df = pd.DataFrame(teams_data, columns=cols)
        # drop duplicates (each game has been counted twice once for every team)
        df = df.drop_duplicates().reset_index(drop=True)
        # sort by epoch time
        df = df.sort_values('Date_int').reset_index(drop=True)
        # convert epoch to UTC time
        df['Date_utc'] = pd.to_datetime(df['Date_int'], unit='ms')
        # convert time to US/Central time-zone
        df['Date_usc'] = (df['Date_utc'].dt.tz_localize(pytz.utc)
                          .dt.tz_convert(us_central))
        # add column with teams abbreviations
        df[['Home_abbr', 'Away_abbr']] = (df[['Home_id', 'Away_id']]
                                          .replace(self.team_id_abbr_dict))
        # add a column with the date only
        df['Date'] = df['Date_usc'].dt.date
        # add week of the year
        df['Week'] = df['Date_usc'].dt.isocalendar().week
        return df

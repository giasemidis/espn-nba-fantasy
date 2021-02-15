import pytz
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from .utils.utils import advanced_stats_by_fantasy_team


class EspnFantasyLeague():
    def __init__(self, league_id, season, n_active_players,
                 url_fantasy, url_nba, cookies):
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
        self.fantasy_stats = ['FT%', 'FG%', 'PTS', '3PM', 'BLK',
                              'STL', 'AST', 'REB', 'TO']

        self.team_id_abbr_dict = {}
        self.team_id_name_dict = {}
        self.team_abbr_name_dict = {}
        self.division_id_name_dict = {}
        self.n_teams = None

        return

    def get_espn_data(self, url_endpoint, endpoints=[], **kargs):
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
        '''
        params = {'view': endpoints, **kargs}
        r = requests.get(url_endpoint, cookies=self.cookies, params=params)
        if r.status_code != 200:
            raise ValueError('Error fetching the teams data')
        data = r.json()
        return data

    def get_teams_data(self):
        '''
        Fetch fantasy teams' data
        '''
        data = self.get_espn_data(self.url_fantasy,
                                  endpoints=['mTeam', 'mRoster', 'mSettings',
                                             'mMatchup'])

        self.get_league_team_division_settings(data)
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

    def make_stat_table(self, data):
        '''
        Makes the *current* total stat table
        It requires the `mTeam` endpoint data
        '''
        stats = [team['valuesByStat'] for team in data['teams']]
        team_abbrs = [team['abbrev'] for team in data['teams']]
        # make dataframe
        stat_table = pd.DataFrame(data=stats, index=team_abbrs)
        # select only headers of interest
        stat_table = stat_table.loc[:, self.stat_id_abbr_dict.keys()]
        # rename stat headers to match the web app
        stat_table.rename(columns=self.stat_id_abbr_dict, inplace=True)
        # update the column data types
        stat_table.astype(self.dtypes)
        return stat_table[self.fantasy_stats]

    def make_standings(self, data, division='all'):
        '''
        Makes the *current* standings table
        It requires the `mTeam` and `mSettings` endpoint data
        '''
        if len(self.division_id_name_dict) == 0:
            self.division_id_name_dict = {
                u['id']: u['name'] for u in
                data['settings']['scheduleSettings']['divisions']
            }
        team_abbrs = [team['abbrev'] for team in data['teams']]
        division_ids = [self.division_id_name_dict[team['divisionId']]
                        for team in data['teams']]
        record = [team['record']['overall'] for team in data['teams']]
        standings = pd.DataFrame(data=record, index=team_abbrs)
        standings = standings.loc[:, ['wins', 'losses', 'ties', 'percentage']]
        standings['Division'] = division_ids

        if len(self.division_id_name_dict) > 1:
            if division.lower() == 'west':
                standings = standings[standings['Division'] == 'West']
            elif division.lower() == 'east':
                standings = standings[standings['Division'] == 'East']

        standings = standings.sort_values('percentage', ascending=False)
        standings['Rank'] = np.arange(1, standings.shape[0] + 1, dtype=int)
        return standings

    def make_stat_standings_table(self, data):
        '''
        Makes the stat and standings table
        '''
        # make the stat table
        stat_table = self.make_stat_table(data)
        # make the standings table
        standings = self.make_standings(data)
        # merge the two on the indices
        table = standings.merge(stat_table, right_index=True, left_index=True)
        return table

    def make_team_scehdule_table(self, data):
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

    def get_fantasy_team_stats_per_round(self, data):
        '''
        Get the total statistics for each fantasy team and each round/week.

        It requires `mTeam`, `mRoster`, `mSettings`, `mMatchup`
        '''
        stat_cols = sorted(self.stat_id_abbr_dict.keys())

        def fun(match, where='home'):
            ''''''
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

        datastore = []
        for i, match in enumerate(data['schedule']):
            if 'home' in match and 'away' in match:
                tmp_home = fun(match, 'home')
                tmp_away = fun(match, 'away')
                if tmp_home is None or tmp_away is None:
                    break
                datastore.append(tmp_home)
                datastore.append(tmp_away)
            else:
                print('Warning, match not found')

        headers = ['Round', 'teamId', 'where', 'wins', 'losses', 'ties']
        cols = headers + stat_cols
        df = pd.DataFrame(datastore, columns=cols)
        df.rename(columns=self.stat_id_abbr_dict, inplace=True)
        df['teamId'] = df['teamId'].replace(self.team_id_abbr_dict)
        df.rename(columns={'teamId': 'teamAbbr'}, inplace=True)

        return df

    def get_total_team_stats_upto_round(self, data, week):
        '''
        Get the total statistics for each fantasy team from the beginning of
        season up to a specified week. If wee = current week, this function
        is equivalent to the `make_stat_table` method above.

        It requires `mTeam`, `mRoster`, `mSettings`, `mMatchup`
        '''
        table = self.get_fantasy_team_stats_per_round(data)
        table = table[table['Round'] <= week]
        stats = list(self.stat_id_abbr_dict.values())
        aggregate = table.groupby('teamAbbr')[stats].sum()
        aggregate['FT%'] = aggregate['FTM'] / aggregate['FTA']
        aggregate['FG%'] = aggregate['FGM'] / aggregate['FGA']
        return aggregate[self.fantasy_stats]

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

    def get_roster_players_mean_stats(self, data, team_abbr,
                                      stat_type_code='002021'):
        '''
        Get the mean stats of all players in the current roster of fantasy team.
        `stat_type_code` can be on of the following:
            * 002021 = season average
            * 012021 = last 7 days average
            * 022021 = last 15 days average
            * 032021 = last 30 days average
            * 102021 = season's projections
            * 002020 = previous season average
        '''
        cols = ['proTeamId', 'injuryStatus'] + list(self.fantasy_stats)

        for teams in data['teams']:
            if teams['abbrev'] == team_abbr:
                team_roster = teams['roster']['entries']
                break

        player_data = []
        for player in team_roster:
            player_name = player['playerPoolEntry']['player']['fullName']
            player_stats = player['playerPoolEntry']['player']['stats']
            injury_status = player['playerPoolEntry']['player']['injuryStatus']
            pro_team_id = player['playerPoolEntry']['player']['proTeamId']
            for stat_type in player_stats:
                if stat_type['id'] == stat_type_code:
                    player_dict = stat_type['averageStats']
                    player_dict['Name'] = player_name
                    player_dict['injuryStatus'] = injury_status
                    player_dict['proTeamId'] = pro_team_id
                    player_data.append(player_dict)
                    break

        df = pd.DataFrame(player_data).set_index('Name').rename(
            columns=self.stat_id_abbr_dict).loc[:, cols]
        return df

    def get_schedule_data(self, since=None, to=None):
        ''''
        Get the agme schedule of NBA team for all players between the
        `since` and `to` dates.
        `since` and `to` can be either None or in date format YYYY-MM-DD.
        '''
        schedule_data = self.get_espn_data(self.url_nba,
                                           endpoints=['proTeamSchedules'])
        us_central = pytz.timezone('US/Central')

        datalist = []
        for proteam in schedule_data['settings']['proTeams']:
            if proteam['abbrev'] == 'FA':
                # skip the free agents
                continue
            row = [proteam['id'], proteam['abbrev'],
                   proteam['location'] + ' ' + proteam['name']]
            # this is the data for each game
            for k, v in proteam['proGamesByScoringPeriod'].items():
                us_datetime = datetime.fromtimestamp(v[0]['date'] // 1000,
                                                     tz=us_central)
                datalist.append(row + [int(k), us_datetime])
        df = pd.DataFrame(datalist,
                          columns=['id', 'abbrev', 'team', 'day', 'date'])
        df = (df.sort_values(
              ['id', 'day', 'date'], ascending=[True, True, True])
              .reset_index(drop=True))
        df['date'] = df['date'].dt.date

        since = df['date'].iloc[0] if since is None else pd.to_datetime(since)
        to = df['date'].iloc[-1] if to is None else pd.to_datetime(to)
        df_period = (df.loc[(df['date'] >= since) & (df['date'] <= to)]
                     .reset_index(drop=True))
        return df_period

    def simulation(self, data, schedule_data, home_team_abbr, away_team_abbr,
                   stat_type_code='002021', n_reps=100_000):
        '''
        Simulates `n_reps` matchups between the home and away fantasy teams.
        The simulations samples the disrete statistical categories (e.g. REB,
        AST, TO, PTS) from a Poisson distribution and the FG% FT% categories
        from a Gaussian distribution.

        Currently, The simualtion does *not* account for
            * inconsistencies such as sampling five 3-pointers and 10 points.
        This is WIP.
        '''

        def merge(team_df, schedule_df):
            merge_df = team_df.reset_index().merge(schedule_df,
                                                   left_on='proTeamId',
                                                   right_on='id')
            merge_df = merge_df.loc[merge_df['injuryStatus'] != 'OUT', :]
            return merge_df

        def simulate_stats(team_abbr):
            team_avg_df = self.get_roster_players_mean_stats(
                data, team_abbr, stat_type_code=stat_type_code)
            merge_df = merge(team_avg_df, schedule_data)

            pois_stats = merge_df.loc[:, poison_stats].values
            gaus_stats = merge_df.loc[:, gaussian_stats].values
            pois = np.random.poisson(lam=pois_stats,
                                     size=(n_reps, *pois_stats.shape))
            pois = np.maximum(0, pois)
            gaus = np.random.normal(loc=gaus_stats, scale=0.2 * gaus_stats,
                                    size=(n_reps, *gaus_stats.shape))
            gaus = np.clip(gaus, 0, 1)
            # total stats for the week
            aggr_pois_stats = pois.sum(axis=1)
            aggr_gaus_stats = gaus.mean(axis=1)
            aggr_stats = np.concatenate((aggr_gaus_stats, aggr_pois_stats),
                                        axis=1)
            return aggr_stats

        gaussian_stats = ['FG%', 'FT%']
        poison_stats = ['3PM', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'TO']

        home_team_sim_stats = simulate_stats(home_team_abbr)
        away_team_sim_stats = simulate_stats(away_team_abbr)

        perc_win = ((home_team_sim_stats > away_team_sim_stats).sum(axis=0)
                    / n_reps * 100)
        perc_win_df = pd.DataFrame(perc_win,
                                   index=gaussian_stats + poison_stats,
                                   columns=['WinProb'])
        # fix for TO
        perc_win_df.loc['TO'] = 100 - perc_win_df.loc['TO']

        return perc_win_df

    def fantasy_team_schedule_count(self, fantasy_team_data, nba_schedule_df,
                                    fantasy_team_abbr):
        # get the roster of the fantasy team
        roster_df = self.get_roster_players_mean_stats(
            fantasy_team_data, fantasy_team_abbr)[['proTeamId', 'injuryStatus']]
        # ingore injured players
        roster_df = roster_df[roster_df['injuryStatus'] != 'OUT']
        # merge based on the pro NBA team ID
        merged_df = roster_df.merge(nba_schedule_df,
                                    left_on='proTeamId', right_on='id')
        # count number of games per day of the fantasy roster team
        groupby_df = (merged_df.groupby(['date']).count()['team']
                      .to_frame('Count'))
        # count valid/active players
        groupby_df['Count_valid'] = np.where(
            groupby_df['Count'] > self.n_active_players, self.n_active_players,
            groupby_df['Count'])
        # make new column to show the unsused subs
        groupby_df['Unused_subs'] = np.where(
            groupby_df['Count'] > self.n_active_players,
            groupby_df['Count'] - self.n_active_players, 0)
        return groupby_df

    def compare_schedules(self, fantasy_team_data, nba_schedule_df,
                          fantasy_team1, fantasy_team2):
        '''
        Compares schedule of a given week of two fantasy teams
        '''
        # get the schedule table of the first team
        table1 = self.fantasy_team_schedule_count(fantasy_team_data,
                                                  nba_schedule_df,
                                                  fantasy_team1)
        # get the schedule table of the second team
        table2 = self.fantasy_team_schedule_count(fantasy_team_data,
                                                  nba_schedule_df,
                                                  fantasy_team2)
        # merge the two on the dates
        table = table1.merge(table2, how='outer',
                             right_index=True, left_index=True,
                             suffixes=('-%s' % fantasy_team1,
                                       '-%s' % fantasy_team2))
        # return the total valid users too.
        total = table.loc[:, table.columns.str.startswith('Count_valid')].sum()
        return table, total

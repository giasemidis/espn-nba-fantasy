import pytz
from datetime import datetime
import numpy as np
import pandas as pd

from .EspnFantasyLeague import EspnFantasyLeague


avg_stats_period_id_dict = {
    '002022': 'season average',
    '012022': 'last 7 days average',
    '022022': 'last 15 days average',
    '032022': 'last 30 days average',
    '102022': "season's projections",
    '002021': 'previous season average'
}


class EspnFantasyMatchUp(EspnFantasyLeague):
    def __init__(self, round, home_team, away_team, league_id, season,
                 n_active_players,
                 url_fantasy, url_nba, cookies, stat_type_code='002022'):
        super().__init__(
            league_id, season, n_active_players, url_fantasy, url_nba, cookies,
            stat_type_code
        )
        self.round = round
        self.home_team = home_team
        self.away_team = away_team

        self.fantasy_teams_data = None
        self.schedule_df = None
        self.n_games_table_df = None

    def get_schedule_data(self, since=None, to=None):  # TODO: Not a method: aux
        ''''
        Get the game schedule of NBA teams for all players between the
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
                datalist.append(row + [int(k), us_datetime,
                                       v[0]['validForLocking']])
        df = pd.DataFrame(datalist,
                          columns=['id', 'abbrev', 'team',
                                   'day', 'date', 'validForLocking'])
        df = (df.sort_values(
              ['id', 'day', 'date'], ascending=[True, True, True])
              .reset_index(drop=True))
        df['date'] = df['date'].dt.date

        since = df['date'].iloc[0] if since is None else pd.to_datetime(since)
        to = df['date'].iloc[-1] if to is None else pd.to_datetime(to)
        period_df = (df.loc[(df['date'] >= since) & (df['date'] <= to)]
                     .reset_index(drop=True))
        # when validForLocking is False, it seems to be for suspended games.
        period_df = period_df[period_df['validForLocking']]
        self.schedule_df = period_df
        return period_df

    def compare_schedules(self, start_date, end_date):  # TODO: Method
        '''
        Compares schedule of a given week of two fantasy teams
        '''
        if self.fantasy_teams_data is None:
            fantasy_team_data = self.get_fantasy_teams_data()
        else:
            fantasy_team_data = self.fantasy_teams_data

        if self.schedule_df is None:
            nba_schedule_df = self.get_schedule_data(start_date, end_date)
        else:
            nba_schedule_df = self.schedule_df

        # get the schedule table of the first team
        table1 = self.fantasy_team_schedule_count(fantasy_team_data,
                                                  nba_schedule_df,
                                                  self.home_team)
        # get the schedule table of the second team
        table2 = self.fantasy_team_schedule_count(fantasy_team_data,
                                                  nba_schedule_df,
                                                  self.away_team)
        # merge the two on the dates
        table = table1.merge(table2, how='outer',
                             right_index=True, left_index=True,
                             suffixes=('-%s' % self.home_team,
                                       '-%s' % self.away_team))
        # return the total valid users too.
        total = table.loc[:, table.columns.str.startswith('Count_valid')].sum()

        self.n_games_table_df = pd.concat(
            (table, total.to_frame('Total').T), axis=0
        )
        return self.n_games_table_df

    def get_total_team_stats_upto_round(self):
        '''
        Get the total statistics for each fantasy team from the beginning of
        season up to a specified week. If wee = current week, this function
        is equivalent to the `make_stat_table` method above.

        It requires `mTeam`, `mRoster`, `mSettings`, `mMatchup`
        '''
        table = self.get_fantasy_team_stats_per_round(data)
        table = table[table['Round'] <= self.round]
        stats = list(self.stat_id_abbr_dict.values())
        aggregate = table.groupby('teamAbbr')[stats].sum()
        aggregate['FT%'] = aggregate['FTM'] / aggregate['FTA']
        aggregate['FG%'] = aggregate['FGM'] / aggregate['FGA']
        return aggregate[self.fantasy_stats]

    def naive_matchup_projection(self):  # TODO: Method
        '''
        Give the projected score of two teams and their projected stats
        based on their *total stats* of the season so far.
        '''
        stats = self.fantasy_stats

        if self.fantasy_teams_data is None:
            fantasy_team_data = self.get_fantasy_teams_data()
        else:
            fantasy_team_data = self.fantasy_teams_data

        df = self.get_total_team_stats_upto_round(
            fantasy_team_data
        )
        # select the two teams' data from the stat table
        if stats is None:
            dftemp = df.loc[[self.home_team, self.away_team], :].T
        else:
            dftemp = df.loc[[self.home_team, self.away_team], stats].T
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
            winner == 0, None,
            np.where(winner == 1, self.home_team, self.away_team)
        )

        print(
            'projected score %s-%s: %d-%d-%d'
            % (self.home_team, self.away_team,
               (winner == 1).sum(), (winner == 2).sum(), (winner == 0).sum())
        )
        return dftemp.round(4).T

    def simulate_projection():  # TODO: Method
        return


    def get_player_stat(self, player_info):
        avg_stats_type = avg_stats_period_id_dict[self.stat_type_code]
        player_name = player_info['fullName']
        player_stats = player_info['stats']
        injury_status = player_info['injuryStatus']
        pro_team_id = player_info['proTeamId']
        player_dict = {}
        for stat_type in player_stats:
            if stat_type['id'] == self.stat_type_code:
                if 'averageStats' not in stat_type:
                    print('Player %s does not have %s available data'
                          % (player_name, avg_stats_type))
                    continue
                player_dict = stat_type['averageStats']
                player_dict['Name'] = player_name
                player_dict['injuryStatus'] = injury_status
                player_dict['proTeamId'] = pro_team_id
                break
        return player_dict

    def get_roster_players_mean_stats(self, data, team_abbr):
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
        cols = ['proTeamId', 'lineupSlotId', 'injuryStatus'] + self.stats_aux

        for teams in data['teams']:
            if teams['abbrev'] == team_abbr:
                team_roster = teams['roster']['entries']
                break

        player_data = []
        for player in team_roster:
            player_info = player['playerPoolEntry']['player']
            # if ((player['lineupSlotId'] == 13)
            #         or (player_info['injuryStatus'] == 'OUT')):
            #     # player is in IR or injured
            #     continue
            player_dict = self.get_player_stat(player_info)
            player_data.append(player_dict)

        df = pd.DataFrame(player_data).set_index('Name').rename(
            columns=self.stat_id_abbr_dict).loc[:, cols]

        return df

    def build_schedule(self, data, team_abbr, remove={}, add={}):
        '''
        Build the team's schedule and stats and apply scenarios by
        removing and adding players.
        '''
        team_avg_df = self.get_roster_players_mean_stats(data, team_abbr)

        if add != {}:
            cols = ['proTeamId', 'injuryStatus'] + self.stats_aux
            players_data = self.get_players_data()
            player_stats_lst = []
            for player_data in players_data['players']:
                if player_data['player']['fullName'] in add.keys():
                    player_avg_stat_dict = self.get_player_stat(
                        player_data['player'])
                    player_stats_lst.append(player_avg_stat_dict)

            df = pd.DataFrame(player_stats_lst).set_index('Name').rename(
                columns=self.stat_id_abbr_dict).loc[:, cols]
            team_avg_df = pd.concat((team_avg_df, df))

        # merge with scedule data
        merge_df = team_avg_df.reset_index().merge(
            schedule_data, left_on='proTeamId', right_on='id'
        )
        # merge_df = merge(team_avg_df, schedule_data)

        rmv_idx = np.zeros(merge_df.shape[0], dtype=bool)
        for player_name, dates in remove.items():
            if dates:
                # ignore players in specific dates
                dates_dt = [pd.Timestamp(date).date() for date in dates]
                rmv_idx |= ((merge_df['Name'] == player_name)
                            & (merge_df['date'].isin(dates_dt)))
            else:
                # ingore all games for the specified players
                rmv_idx |= (merge_df['Name'] == player_name)
        for player_name, dates in add.items():
            # ignore dates not considered from the added players
            if dates:
                dates_dt = [pd.Timestamp(date).date() for date in dates]
                rmv_idx |= ((merge_df['Name'] == player_name)
                            & (~merge_df['date'].isin(dates_dt)))

        merge_df = merge_df[~rmv_idx]

        return merge_df

    def simulation(self, data, schedule_data, home_team_abbr, away_team_abbr,
                   n_reps=100_000,
                   current_score=None,
                   ignore_players={'home': {}, 'away': {}},
                   add_players={'home': {}, 'away': {}}):
        '''
        Simulates `n_reps` matchups between the home and away fantasy teams.
        The simulation samples the disrete statistical categories (FGA, FTA,
        3PM, REB, AST, STL, BLK, TO, PTS) from a Poisson distribution.
        FGM and FTM are estimated from the simulated attempts (bounded above)
        and the average respective percentage (FG%, FT%). From made and attempts
        the aggregated percentage is estimated.

        Currently, The simualtion does *not* account for
            * inconsistencies such as sampling five 3-pointers and 10 points.
        This is WIP.
        '''

        print('Player stats type %s'
              % avg_stats_period_id_dict[self.stat_type_code])

        fga_idx = self.simulation_stats.index('FGA')
        fta_idx = self.simulation_stats.index('FTA')
        fgm_idx = self.simulation_stats.index('FGM')
        ftm_idx = self.simulation_stats.index('FTM')

        home_team_sim_stats = build_and_simulate(home_team_abbr,
                                                 ignore_players['home'],
                                                 add_players['home'])
        away_team_sim_stats = build_and_simulate(away_team_abbr,
                                                 ignore_players['away'],
                                                 add_players['away'])

        if current_score is not None:
            home_team_sim_stats = (
                current_score.loc[home_team_abbr, self.simulation_stats].values
                + home_team_sim_stats
            )
            away_team_sim_stats = (
                current_score.loc[away_team_abbr, self.simulation_stats].values
                + away_team_sim_stats
            )

        hm_fg = shot_percentage(home_team_sim_stats, fgm_idx, fga_idx)
        hm_ft = shot_percentage(home_team_sim_stats, ftm_idx, fta_idx)
        aw_fg = shot_percentage(away_team_sim_stats, fgm_idx, fga_idx)
        aw_ft = shot_percentage(away_team_sim_stats, ftm_idx, fta_idx)

        stats_idx = np.isin(self.simulation_stats, self.fantasy_stats)
        home_team_sim_9_stats = np.concatenate(
            (hm_fg, hm_ft, home_team_sim_stats[:, stats_idx]), axis=1)
        away_team_sim_9_stats = np.concatenate(
            (aw_fg, aw_ft, away_team_sim_stats[:, stats_idx]), axis=1)

        home_mean_sim_stas = home_team_sim_9_stats.mean(axis=0)
        away_mean_sim_stas = away_team_sim_9_stats.mean(axis=0)

        mean_stats_matchup_df = pd.DataFrame({
            home_team_abbr: home_mean_sim_stas,
            away_team_abbr: away_mean_sim_stas
        }, index=self.fantasy_stats)

        perc_win = ((home_team_sim_9_stats > away_team_sim_9_stats).sum(axis=0)
                    / n_reps * 100)
        perc_win_df = pd.DataFrame(perc_win, index=self.fantasy_stats,
                                   columns=['HomeWinProb'])
        # fix for TO
        perc_win_df.loc['TO'] = 100 - perc_win_df.loc['TO']

        # merge the two.
        out_df = pd.merge(perc_win_df, mean_stats_matchup_df,
                          left_index=True, right_index=True)

        return out_df

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

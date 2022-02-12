import pytz
from datetime import datetime
import numpy as np
import pandas as pd

from .EspnFantasyLeague import EspnFantasyLeague
from .utils.utils import (
    extract_player_stat,
    fantasy_team_schedule_count,
    filter_by_round_team_stats,
    simulate_schedule,
    shot_percentage
)

STAT_PERIOD_DICT = {
    '002022': 'season average',
    '012022': 'last 7 days average',
    '022022': 'last 15 days average',
    '032022': 'last 30 days average',
    '102022': "season's projections",
    '002021': 'previous season average'
}

SCN_DEFAULT = {"remove": {}, "add": {}}


class EspnFantasyMatchUp(EspnFantasyLeague):
    def __init__(self, round, home_team, away_team, start_date, end_date,
                 league_id, season, n_active_players,
                 url_fantasy, url_nba, cookies, stat_type_code='002022',
                 home_scn_players=SCN_DEFAULT, away_scn_players=SCN_DEFAULT):
        super().__init__(
            league_id, season, n_active_players, url_fantasy, url_nba, cookies,
            stat_type_code
        )
        self.round = round
        self.home_team = home_team
        self.away_team = away_team
        self.start_date = start_date
        self.end_date = end_date

        self.fantasy_teams_data = None
        self.nba_schedule_df = None
        self.n_games_table_df = None
        self.current_score_df = None
        self.simulation_result = None
        setattr(self, f"{home_team}_roster_stats", None)
        setattr(self, f"{away_team}_roster_stats", None)
        setattr(self, f"{home_team}_schedule_stats", None)
        setattr(self, f"{away_team}_schedule_stats", None)
        setattr(self, f"{home_team}_scn_players", home_scn_players)
        setattr(self, f"{away_team}_scn_players", away_scn_players)

    def h2h_season_stats_comparison(self):
        '''
        Give the projected score of two teams and their projected stats
        based on their *total stats* of the season so far.
        '''
        df = self.get_total_team_stats_upto_round(week=self.round)
        # select the two teams' data from the stat table
        dftemp = df.loc[[self.home_team, self.away_team], :].T

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

        print('projected score %s-%s: %d-%d-%d' % (
            self.home_team, self.away_team,
            (winner == 1).sum(),
            (winner == 2).sum(),
            (winner == 0).sum())
        )
        return dftemp.round(4)

    def get_current_score(self):
        """
        TODO
        """
        if (self.current_score_df is not None):
            return self.current_score_df

        table = self.get_fantasy_team_stats_per_round()
        extra_stats = [
            s for s in self.simulation_stats if s not in self.fantasy_stats
        ]
        stats = self.fantasy_stats + extra_stats
        teams = [self.home_team, self.away_team]
        current_score = filter_by_round_team_stats(
            table, self.round, stats, teams).T

        self.current_score_df = current_score
        return current_score

    def get_schedule_data(self):
        ''''
        Get the game schedule of NBA teams for all players between the
        `self.start_date` and `self.end_date` dates.
        The dates can be either None or in date format YYYY-MM-DD.
        '''

        if self.nba_schedule_df is not None:
            return self.nba_schedule_df

        schedule_data = self.get_espn_data(
            self.url_nba, endpoints=['proTeamSchedules']
        )
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

        since = (
            df['date'].iloc[0] if self.start_date is None
            else pd.to_datetime(self.start_date)
        )
        to = (
            df['date'].iloc[-1] if self.end_date is None
            else pd.to_datetime(self.end_date)
        )
        period_df = (df.loc[(df['date'] >= since) & (df['date'] <= to)]
                     .reset_index(drop=True))

        # when validForLocking is False, it seems to be for suspended games.
        period_df = period_df[period_df['validForLocking']]
        self.nba_schedule_df = period_df
        return period_df

    def extract_roster_mean_stats(self, team_abbr):
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

        team_roster_stats = getattr(self, f"{team_abbr}_roster_stats")
        if team_roster_stats is not None:
            return team_roster_stats

        data = self.get_fantasy_teams_data()

        cols = ['proTeamId', 'lineupSlotId', 'injuryStatus'] + self.stats_aux

        for teams in data['teams']:
            if teams['abbrev'] == team_abbr:
                team_roster = teams['roster']['entries']
                break

        player_data = []
        for player in team_roster:
            player_info = player['playerPoolEntry']['player']
            player_dict = extract_player_stat(
                player_info, self.stat_type_code)
            player_dict.update({"lineupSlotId": player['lineupSlotId']})
            player_data.append(player_dict)

        df = pd.DataFrame(player_data).set_index('Name').rename(
            columns=self.stat_id_abbr_dict).loc[:, cols]

        setattr(self, f"{team_abbr}_roster_stats", df)

        return df

    def team_shcedule_df(self, team):
        """
        TODO
        """
        schedule_stats = getattr(self, f"{team}_schedule_stats")
        if schedule_stats is not None:
            return schedule_stats

        nba_schedule_df = self.get_schedule_data()
        roster_df = self.extract_roster_mean_stats(team)
        # merge based on the pro NBA team ID
        merged_df = (
            roster_df.reset_index()
            .merge(nba_schedule_df, left_on='proTeamId', right_on='id')
            .set_index("Name")
        )

        added_players_df = self.add_scn_players(team)
        merged_df = pd.concat((merged_df, added_players_df), axis=0)
        merged_df = self.remove_scn_players(merged_df, team)

        # ignore injured players or players in the IR list
        keep_indx = (merged_df['injuryStatus'] != 'OUT') & \
            (merged_df['lineupSlotId'] != 13)
        merged_df = merged_df[keep_indx]

        setattr(self, f"{team}_schedule_stats", merged_df)
        return merged_df

    def add_scn_players(self, team):
        cols = ['Name', 'proTeamId', 'injuryStatus'] + self.stats_aux
        add = getattr(self, f"{team}_scn_players")["add"]
        out_data = []
        if add:
            nba_schedule_df = self.get_schedule_data()
            players_data = self.get_players_data()
            player_stats_lst = []
            for player_name, dates in add.items():
                for player_data in players_data['players']:
                    if player_data['player']['fullName'] in player_name:
                        player_avg_stat_dict = extract_player_stat(
                            player_data['player'], self.stat_type_code)
                        player_stats_lst.append(player_avg_stat_dict)
                        # print(player_avg_stat_dict)
                        break
                else:
                    print(f"Player {player_name} not found")

                stat_df = (
                    pd.DataFrame(player_stats_lst)
                    .rename(columns=self.stat_id_abbr_dict).loc[:, cols]
                )
                if dates:
                    date_mask = nba_schedule_df["date"].isin(dates)
                    nba_schedule_df = nba_schedule_df[date_mask]
                merged_df = (
                    stat_df.merge(
                        nba_schedule_df, left_on='proTeamId', right_on='id'
                    ).set_index('Name')
                )
                out_data.append(merged_df)
            df = pd.concat(out_data, axis=0)
        else:
            df = pd.DataFrame([])
        return df

    def remove_scn_players(self, schedule_stats, team):
        # Now remove players unwanted players dates
        rmv_idx = np.zeros(schedule_stats.shape[0], dtype=bool)
        remove = getattr(self, f"{team}_scn_players")["remove"]
        for player_name, dates in remove.items():
            if dates:
                # ignore players in specific dates
                dates_dt = [pd.Timestamp(date).date() for date in dates]
                rmv_idx |= ((schedule_stats.index == player_name)
                            & (schedule_stats['date'].isin(dates_dt)))
            else:
                # ingore all games for the specified players
                rmv_idx |= (schedule_stats.index == player_name)
        schedule_stats = schedule_stats[~rmv_idx]
        return schedule_stats

    def compare_schedules(self):
        '''
        Compares schedule of a given week of two fantasy teams
        '''

        if self.n_games_table_df is not None:
            return self.n_games_table_df

        home_team_schedule = self.team_shcedule_df(self.home_team)
        away_team_schedule = self.team_shcedule_df(self.away_team)

        # get the schedule table of the first team
        home_table = fantasy_team_schedule_count(
            home_team_schedule, self.n_active_players
        )
        # get the schedule table of the second team
        away_table = fantasy_team_schedule_count(
            away_team_schedule, self.n_active_players
        )
        # merge the two on the dates
        table = home_table.merge(
            away_table, how='outer', right_index=True, left_index=True,
            suffixes=('-%s' % self.home_team, '-%s' % self.away_team)
        )
        # return the total valid users too.
        total = table.loc[:, table.columns.str.startswith('Count_valid')].sum()

        self.n_games_table_df = pd.concat(
            (table, total.to_frame('Total').T), axis=0
        )
        return self.n_games_table_df

    def simulation(self, n_reps=100_000, use_current_score=False):
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
              % STAT_PERIOD_DICT[self.stat_type_code])

        fga_idx = self.simulation_stats.index('FGA')
        fta_idx = self.simulation_stats.index('FTA')
        fgm_idx = self.simulation_stats.index('FGM')
        ftm_idx = self.simulation_stats.index('FTM')

        home_schedule_stats = self.team_shcedule_df(self.home_team)
        away_schedule_stats = self.team_shcedule_df(self.away_team)
        home_team_sim_stats = simulate_schedule(
            home_schedule_stats, self.poisson_stats, n_reps)
        away_team_sim_stats = simulate_schedule(
            away_schedule_stats, self.poisson_stats, n_reps)

        if use_current_score:
            current_score = self.get_current_score().T
            home_team_sim_stats = (
                current_score.loc[self.home_team, self.simulation_stats].values
                + home_team_sim_stats
            )
            away_team_sim_stats = (
                current_score.loc[self.away_team, self.simulation_stats].values
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
            self.home_team: home_mean_sim_stas,
            self.away_team: away_mean_sim_stas
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

        self.simulation_result = out_df
        return out_df
